#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"

#include <cstdint>
#include <string>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/RotationUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace tensor_ext {

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"

template <typename T>
static SmallVector<T> rotateAttrValues(SmallVector<T> elts, int64_t shift) {
  SmallVector<T> newElts;
  newElts.reserve(elts.size());
  for (int i = 0; i < elts.size(); ++i) {
    int64_t source = (i + shift + elts.size()) % elts.size();
    newElts.push_back(elts[source]);
  }
  return newElts;
}

// tensor_ext.rotate (arith.constant dense) -> arith.constant (rotated dense)
struct FoldRotateOfConst : public OpRewritePattern<tensor_ext::RotateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor_ext::RotateOp op,
                                PatternRewriter& rewriter) const override {
    auto constantOp = op.getTensor().getDefiningOp<arith::ConstantOp>();
    if (!constantOp)
      return rewriter.notifyMatchFailure(op, "requires constant operand");

    // Could be smarter here by using SCCP analysis, though most cases will not
    // need it because the shift will itself be SCCP'd to a constant.
    Attribute shiftConst =
        dyn_cast<Attribute>(getAsOpFoldResult(op.getShift()));
    if (!shiftConst)
      return rewriter.notifyMatchFailure(
          op, "requires constant shift OpFoldResult");
    int64_t shift = cast<IntegerAttr>(shiftConst).getInt();

    arith::ConstantOp newOp;

    DenseIntElementsAttr denseIntAttr =
        dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr());
    if (denseIntAttr) {
      SmallVector<APInt> elts(denseIntAttr.getValues<APInt>());
      SmallVector<APInt> newElts = rotateAttrValues<APInt>(elts, shift);
      DenseIntElementsAttr newAttr =
          DenseIntElementsAttr::get(denseIntAttr.getType(), newElts);
      newOp = arith::ConstantOp::create(
          rewriter, op.getLoc(), constantOp.getResult().getType(), newAttr);
    }

    DenseFPElementsAttr denseFloatAttr =
        dyn_cast<DenseFPElementsAttr>(constantOp.getValueAttr());
    if (denseFloatAttr) {
      SmallVector<APFloat> elts(denseFloatAttr.getValues<APFloat>());
      SmallVector<APFloat> newElts = rotateAttrValues<APFloat>(elts, shift);
      DenseFPElementsAttr newAttr =
          DenseFPElementsAttr::get(denseFloatAttr.getType(), newElts);
      newOp = arith::ConstantOp::create(
          rewriter, op.getLoc(), constantOp.getResult().getType(), newAttr);
    }

    if (!newOp)
      return rewriter.notifyMatchFailure(
          op, "requires constant operand with dense int or float attribute");

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

void RotateOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<FoldRotateOfConst>(context);
  populateWithGenerated(results);
}

LogicalResult RotateOp::verify() {
  auto x = getTensor().getType();
  if (x.getRank() < 1) {
    return emitOpError() << "requires operand rank >= 1";
  }
  return success();
}

::llvm::SmallVector<::mlir::OpFoldResult> RotateOp::getRotationIndices() {
  return {getShift()};
}

::llvm::SmallVector<::mlir::OpFoldResult>
RotateAndReduceOp::getRotationIndices() {
  int64_t period = getPeriod().getZExtValue();
  int64_t steps = getSteps().getZExtValue();
  bool hasPlaintexts = static_cast<bool>(getPlaintexts());
  auto indices = rotateAndReduceRotationIndices(period, steps, hasPlaintexts);
  SmallVector<OpFoldResult> result;
  auto* ctx = getContext();
  for (int64_t idx : indices) {
    result.push_back(IntegerAttr::get(IndexType::get(ctx), idx));
  }
  return result;
}

LogicalResult verifyLayoutMatchesType(const Attribute& layoutAttr, Type type,
                                      Operation* op) {
  auto shapedType = dyn_cast<ShapedType>(type);

  if (auto newLayout = dyn_cast<LayoutAttr>(layoutAttr)) {
    presburger::IntegerRelation rel = newLayout.getIntegerRelation();
    if (shapedType && rel.getNumDomainVars() != shapedType.getRank()) {
      return op->emitOpError()
             << "requires tensor rank to match the layout's domain size, "
                "but found rank "
             << shapedType.getRank() << " and domain size "
             << rel.getNumDomainVars();
    }
    return success();
  }

  if (auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(layoutAttr)) {
    // Assert the attr has shape <N x 4>
    int64_t rank = denseElementsAttr.getType().getRank();
    if (rank != 2)
      return op->emitOpError()
             << "requires permutation attribute to Rank 2, but "
             << "found shape <" << denseElementsAttr.getType() << ">";

    int64_t cols = denseElementsAttr.getType().getDimSize(1);
    if (cols != 4)
      return op->emitOpError()
             << "requires permutation attribute to be of shape <N x 4>, but "
                "found shape <"
             << denseElementsAttr.getType() << ">" << "Rank: " << rank
             << " Cols: " << cols << "\n";
    return success();
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(layoutAttr)) {
    if (arrayAttr.empty()) {
      return op->emitOpError() << "layout array cannot be empty";
    }
    return verifyLayoutMatchesType(arrayAttr[0], type, op);
  }

  return op->emitOpError("Unsupported layout attribute");
}

LogicalResult verifyCompositeLayout(ArrayAttr compositeLayout, Operation* op) {
  if (compositeLayout.empty()) {
    return op->emitOpError() << "layout array cannot be empty";
  }

  // Verify that the domain of each layout matches the range of the previous.
  auto firstLayout = dyn_cast<LayoutAttr>(compositeLayout[0]);
  if (!firstLayout) {
    return op->emitOpError() << "first layout in array must be a LayoutAttr";
  }

  for (unsigned i = 1; i < compositeLayout.size(); ++i) {
    auto prevLayout = dyn_cast<LayoutAttr>(compositeLayout[i - 1]);
    auto currLayout = dyn_cast<LayoutAttr>(compositeLayout[i]);
    if (!prevLayout || !currLayout) {
      return op->emitOpError() << "all layouts in the array must be LayoutAttr";
    }

    // Verify that the upper and lower bounds of all range vars are
    // non-negative.
    for (unsigned i = 0; i < prevLayout.getIntegerRelation().getNumRangeVars();
         ++i) {
      auto lb = prevLayout.getIntegerRelation().getConstantBound64(
          presburger::BoundType::LB,
          prevLayout.getIntegerRelation().getVarKindOffset(
              presburger::VarKind::Range) +
              i);
      if (!lb.has_value() || *lb < 0) {
        return op->emitOpError()
               << "Negative lower bound in range variable " << i << ": "
               << (lb.has_value() ? std::to_string(*lb) : "null");
      }
      auto ub = prevLayout.getIntegerRelation().getConstantBound64(
          presburger::BoundType::UB,
          prevLayout.getIntegerRelation().getVarKindOffset(
              presburger::VarKind::Range) +
              i);
      if (!ub.has_value() || *ub < 0) {
        return op->emitOpError()
               << "Negative upper bound in range variable " << i << ": "
               << (ub.has_value() ? std::to_string(*ub) : "null");
      }
    }

    if (currLayout.getIntegerRelation().getNumDomainVars() !=
        prevLayout.getIntegerRelation().getNumRangeVars()) {
      return op->emitOpError()
             << "layout " << i << " domain size ("
             << currLayout.getIntegerRelation().getNumDomainVars()
             << ") must match layout " << (i - 1) << " range size ("
             << prevLayout.getIntegerRelation().getNumRangeVars() << ")";
    }
  }
  return success();
}

OpFoldResult ConvertLayoutOp::fold(FoldAdaptor adaptor) {
  auto tensor = getValue();
  auto fromLayout = getFromLayout();
  auto toLayout = getToLayout();

  if (fromLayout == toLayout) {
    return tensor;
  }

  auto inputOp = tensor.getDefiningOp<ConvertLayoutOp>();
  if (!inputOp || toLayout != inputOp.getFromLayout()) return OpFoldResult();

  return inputOp.getValue();
}

LogicalResult ConvertLayoutOp::verify() {
  LogicalResult inputVerification =
      verifyLayoutMatchesType(getFromLayout(), getValue().getType(), *this);
  if (failed(inputVerification)) {
    return inputVerification;
  }

  LogicalResult outputVerification =
      verifyLayoutMatchesType(getToLayout(), getResult().getType(), *this);
  if (failed(outputVerification)) {
    return outputVerification;
  }

  if (auto compositeLayout = dyn_cast<ArrayAttr>(getFromLayout())) {
    if (failed(verifyCompositeLayout(compositeLayout, *this))) {
      return failure();
    }
  }

  if (auto compositeLayout = dyn_cast<ArrayAttr>(getToLayout())) {
    // Verify first layout domain matches input type.
    if (failed(verifyLayoutMatchesType(compositeLayout[0], getValue().getType(),
                                       *this))) {
      return failure();
    }
    if (failed(verifyCompositeLayout(compositeLayout, *this))) {
      return failure();
    }
  }

  return success();
}

LogicalResult AssignLayoutOp::verify() {
  Attribute layoutAttr = getLayout();
  Type inputType = getValue().getType();

  if (auto arrayAttr = dyn_cast<ArrayAttr>(layoutAttr)) {
    if (!getDomainSchedule().empty()) {
      return emitOpError() << "domainSchedule is not supported for composite "
                              "layouts";
    }

    if (arrayAttr.empty()) {
      return emitOpError() << "layout array cannot be empty";
    }

    // Verify first layout domain matches input type.
    if (failed(verifyLayoutMatchesType(arrayAttr[0], inputType, *this))) {
      return failure();
    }

    if (failed(verifyCompositeLayout(arrayAttr, *this))) {
      return failure();
    }

  } else {
    if (failed(verifyLayoutMatchesType(layoutAttr, inputType, *this))) {
      return failure();
    }
  }

  if (!getDomainSchedule().empty()) {
    // Domain schedules are unexpected for composite layouts and permutations.
    LayoutAttr layout = dyn_cast<LayoutAttr>(getLayout());

    if (!layout) {
      return emitOpError()
             << "requires LayoutAttr when domainSchedule is provided";
    }
    presburger::IntegerRelation rel = layout.getIntegerRelation();
    for (int64_t idx : getDomainSchedule()) {
      if (idx < 0 || idx >= rel.getNumDomainVars()) {
        return emitOpError()
               << "domainSchedule index " << idx << " is out of bounds [0, "
               << rel.getNumDomainVars() << ")";
      }
    }
  }

  return success();
}

LogicalResult UnpackOp::verify() {
  return verifyLayoutMatchesType(getLayout(), getResult().getType(), *this);
}

LogicalResult RemapOp::verify() {
  auto tensorTy = getInput().getType();
  if (tensorTy.getRank() != 2) {
    return emitOpError() << "requires input tensor to be rank 2 "
                         << "(ciphertext, slot), but found rank "
                         << tensorTy.getRank();
  }

  if (isa<LayoutAttr>(getPermutation())) {
    return success();
  }

  if (auto denseElementsAttr =
          dyn_cast<DenseIntElementsAttr>(getPermutation())) {
    // Assert the attr has shape <N x 4>
    int64_t rank = denseElementsAttr.getType().getRank();
    int64_t cols = denseElementsAttr.getType().getDimSize(1);
    if (rank != 2 || cols != 4) {
      return emitOpError()
             << "requires permutation attribute to be of shape <N x 4>, but "
                "found shape <"
             << denseElementsAttr.getType() << ">";
    }
  }

  return success();
}

LogicalResult RotateAndReduceOp::verify() {
  auto numSteps = getSteps().getZExtValue();

  auto x = getTensor().getType();
  // TODO(#924): Currently RotateAndReduceOp only supports rotating a 1-D
  // vector, or a vector with only one non-unit dimension that is treated as
  // the major dimension.
  if (x.getRank() != 1) {
    if (llvm::count_if(x.getShape(), [](auto dim) { return dim != 1; }) != 1) {
      return emitOpError() << "requires a 1-D input tensor or tensor with "
                              "single non-unit dimension, but found "
                           << x;
    }
  }
  // Final dimension is the number of slots
  if (numSteps > x.getDimSize(x.getRank() - 1)) {
    return emitOpError()
           << "requires steps to be less than or equal "
              "to the input tensor's dimension, but found reductions="
           << numSteps
           << " and tensor dimension=" << x.getDimSize(x.getRank() - 1);
  }

  if (getPlaintexts()) {
    auto numPlaintexts = getPlaintexts().getType().getDimSize(0);
    if (numPlaintexts != numSteps) {
      return emitOpError()
             << "requires plaintext tensor to have the same number of "
                "elements as steps, but found numPlaintexts="
             << numPlaintexts << " and steps=" << numSteps;
    }
  }

  auto period = getPeriod().getZExtValue();
  if (period <= 0 || period > x.getNumElements()) {
    return emitOpError() << "requires period to be within the range of the "
                            "tensor, but found period "
                         << period << " and tensor with " << x.getNumElements()
                         << " elements";
  }

  return success();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
