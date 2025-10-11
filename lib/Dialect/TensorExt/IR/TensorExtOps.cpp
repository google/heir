#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"

#include <cstdint>
#include <string>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/AffineMapUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
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

  return op->emitOpError("Unsupported layout attribute");
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

  return success();
}

LogicalResult AssignLayoutOp::verify() {
  return verifyLayoutMatchesType(getLayout(), getValue().getType(), *this);
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
  // vector, or a vector with only one non-unit dimension that is treated as the
  // major dimension.
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
