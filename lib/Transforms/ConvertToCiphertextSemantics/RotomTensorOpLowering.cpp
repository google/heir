#include "lib/Transforms/ConvertToCiphertextSemantics/RotomTensorOpLowering.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/MapVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using tensor_ext::LayoutAttr;

static auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
static auto& kMaterializedAttrName = "tensor_ext.layout_materialized";

static void setMaterializedAttr(Operation* op) {
  op->setAttr(kMaterializedAttrName, UnitAttr::get(op->getContext()));
}

LayoutAttr RotomTensorOpLowering::getLayoutAttr(Value value) const {
  auto layoutLookup = typeConverter->getContextualAttr(value);
  if (failed(layoutLookup)) {
    return nullptr;
  }
  return dyn_cast<LayoutAttr>(layoutLookup.value());
}

// A plaintext 1/0 mask selecting `slots` of a single-ciphertext row.
static Value buildSlotMask(ImplicitLocOpBuilder& b, RankedTensorType rowType,
                           ArrayRef<int64_t> slots) {
  Type elementType = rowType.getElementType();
  Attribute zero, one;
  if (isa<FloatType>(elementType)) {
    zero = FloatAttr::get(elementType, 0.0);
    one = FloatAttr::get(elementType, 1.0);
  } else {
    zero = IntegerAttr::get(elementType, 0);
    one = IntegerAttr::get(elementType, 1);
  }
  SmallVector<Attribute> values(rowType.getNumElements(), zero);
  for (int64_t slot : slots) values[slot] = one;
  auto dense = DenseElementsAttr::get(rowType, values);
  auto mask = arith::ConstantOp::create(b, rowType, dense);
  setMaterializedAttr(mask);
  return mask;
}

Value RotomTensorOpLowering::convertToLayout(ImplicitLocOpBuilder& b,
                                             Value value, LayoutAttr fromLayout,
                                             LayoutAttr toLayout) const {
  if (fromLayout == toLayout) return value;
  auto sourceType = dyn_cast<RankedTensorType>(value.getType());
  if (!sourceType || sourceType.getRank() != 2) return nullptr;
  const int64_t numSlots = sourceType.getDimSize(1);

  IntegerRelation toRelation = toLayout.getIntegerRelation();
  auto ctUpperBound = toRelation.getConstantBound64(
      presburger::BoundType::UB,
      toRelation.getVarKindOffset(presburger::VarKind::Range));
  if (!ctUpperBound) return nullptr;
  const int64_t numTargetCt = *ctUpperBound + 1;

  SmallVector<rotom::LayoutExpansionStep> steps;
  if (numTargetCt == sourceType.getDimSize(0)) {
    // Same ciphertext count: the same cheapest-route choice the layout
    // assignment priced -- a convert_layout for the shift network, or the
    // explicit steps below when they need fewer rotations.
    FailureOr<rotom::SameCountConversionChoice> choice =
        rotom::chooseSameCountConversion(fromLayout, toLayout, numSlots);
    if (failed(choice)) return nullptr;
    if (!choice->useSteps) {
      Value v =
          tensor_ext::ConvertLayoutOp::create(b, value, fromLayout, toLayout);
      setAttributeAssociatedWith(v, kLayoutAttrName, toLayout);
      return v;
    }
    steps = std::move(choice->steps);
  } else {
    FailureOr<SmallVector<rotom::LayoutExpansionStep>> planned =
        rotom::planLayoutExpansion(fromLayout.getIntegerRelation(), toRelation,
                                   numSlots);
    if (failed(planned)) return nullptr;
    steps = std::move(*planned);
  }
  auto rowType =
      RankedTensorType::get({1, numSlots}, sourceType.getElementType());

  SmallVector<Value> rows(numTargetCt, Value());
  SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1),
                                     b.getIndexAttr(numSlots)};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
  // Steps sharing (source ciphertext, shift) share one rotated row; reusing
  // it across every target it feeds keeps the emitted rotation count equal
  // to the priced one (a baby-step expansion rotates B-1 times, not once
  // per replicated block).
  DenseMap<std::pair<int64_t, int64_t>, Value> rotatedRows;
  for (const rotom::LayoutExpansionStep& step : steps) {
    if (step.targetCt < 0 || step.targetCt >= numTargetCt) return nullptr;
    Value& row = rotatedRows[{step.sourceCt, step.shift}];
    if (!row) {
      SmallVector<OpFoldResult> offsets = {b.getIndexAttr(step.sourceCt),
                                           b.getIndexAttr(0)};
      row = tensor::ExtractSliceOp::create(b, rowType, value, offsets, sizes,
                                           strides);
      setMaterializedAttr(row.getDefiningOp());
      if (step.shift != 0) {
        Value shift = arith::ConstantIndexOp::create(b, step.shift);
        setMaterializedAttr(shift.getDefiningOp());
        row = tensor_ext::RotateOp::create(b, row, shift);
        setMaterializedAttr(row.getDefiningOp());
      }
    }
    // The mask is per step (target slots differ), so it must not touch the
    // shared cached row.
    Value stepRow = row;
    if (static_cast<int64_t>(step.targetSlots.size()) != numSlots) {
      Value mask = buildSlotMask(b, rowType, step.targetSlots);
      Operation* masked =
          makeAppropriatelyTypedMulOp(b, b.getLoc(), stepRow, mask);
      setMaterializedAttr(masked);
      stepRow = masked->getResult(0);
    }
    Value& target = rows[step.targetCt];
    if (!target) {
      target = stepRow;
    } else {
      Operation* add =
          makeAppropriatelyTypedAddOp(b, b.getLoc(), target, stepRow);
      setMaterializedAttr(add);
      target = add->getResult(0);
    }
  }
  for (Value row : rows) {
    if (!row) return nullptr;
  }

  Value result;
  if (numTargetCt == 1) {
    result = rows[0];
  } else {
    auto concat = tensor::ConcatOp::create(b, /*dim=*/0, rows);
    setMaterializedAttr(concat);
    result = concat.getResult();
  }
  Operation* def = result.getDefiningOp();
  def->setAttr(kLayoutAttrName, toLayout);
  setMaterializedAttr(def);
  setAttributeAssociatedWith(result, kLayoutAttrName, toLayout);
  return result;
}

LogicalResult RotomTensorOpLowering::lowerElementwiseBinary(
    Operation* op, Value originalResult, ValueRange adaptorOperands,
    ContextAwareConversionPatternRewriter& rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "Converting elementwise op with Rotom kernel: "
                          << *op << "\n");

  if (op->getNumOperands() != 2 || op->getNumResults() != 1 ||
      adaptorOperands.size() != 2) {
    return rewriter.notifyMatchFailure(
        op, "Rotom elementwise lowering requires a binary single-result op");
  }

  auto resultType = dyn_cast<RankedTensorType>(originalResult.getType());
  if (!resultType || !resultType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        op, "Rotom elementwise lowering requires a static tensor result");
  }

  auto lhsType = dyn_cast<RankedTensorType>(adaptorOperands[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(adaptorOperands[1].getType());
  if (!lhsType || !rhsType) {
    return rewriter.notifyMatchFailure(
        op, "Rotom elementwise lowering requires tensor operands");
  }

  LayoutAttr lhsLayout = getLayoutAttr(adaptorOperands[0]);
  LayoutAttr rhsLayout = getLayoutAttr(adaptorOperands[1]);
  auto outputLayout = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
  if (!lhsLayout || !rhsLayout || !outputLayout) {
    return rewriter.notifyMatchFailure(
        op, "missing tensor_ext.layout attributes for Rotom elementwise op");
  }

  Type convertedResultType =
      typeConverter->convertType(resultType, outputLayout);
  if (!convertedResultType) {
    return rewriter.notifyMatchFailure(
        op, "failed to convert Rotom elementwise result type");
  }
  // Operand ciphertext counts may differ from the output's (the conversion
  // below expands or compacts them); only the slot count and element type
  // must agree.
  auto ciphertextSemanticType = dyn_cast<RankedTensorType>(convertedResultType);
  if (!ciphertextSemanticType || ciphertextSemanticType.getRank() != 2 ||
      lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      lhsType.getDimSize(1) != ciphertextSemanticType.getDimSize(1) ||
      rhsType.getDimSize(1) != ciphertextSemanticType.getDimSize(1) ||
      lhsType.getElementType() != ciphertextSemanticType.getElementType() ||
      rhsType.getElementType() != ciphertextSemanticType.getElementType()) {
    return rewriter.notifyMatchFailure(
        op,
        "Rotom elementwise lowering requires ciphertext semantic tensors "
        "with matching slot counts and element types");
  }

  bool isAdd = isa<arith::AddFOp, arith::AddIOp>(op);
  bool isSub = isa<arith::SubFOp, arith::SubIOp>(op);
  bool isMul = isa<arith::MulFOp, arith::MulIOp>(op);
  if (!isAdd && !isSub && !isMul) {
    return rewriter.notifyMatchFailure(op,
                                       "unsupported Rotom elementwise op kind");
  }

  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  rewriter.setInsertionPointAfter(op);

  // Convert-then-compute: bring each operand to the shared output layout (a
  // no-op when it already is), then perform the elementwise op once at that
  // layout. A same-ciphertext-count conversion becomes a convert_layout the
  // implement-shift-network pass lowers into rotations + masks; a
  // count-changing one is emitted as explicit rotate/mask/accumulate steps.
  Value lhs = convertToLayout(b, adaptorOperands[0], lhsLayout, outputLayout);
  Value rhs = convertToLayout(b, adaptorOperands[1], rhsLayout, outputLayout);
  if (!lhs || !rhs) {
    return rewriter.notifyMatchFailure(
        op, "failed to convert a Rotom elementwise operand layout");
  }

  Operation* result =
      isAdd   ? makeAppropriatelyTypedAddOp(b, op->getLoc(), lhs, rhs)
      : isSub ? makeAppropriatelyTypedSubOp(b, op->getLoc(), lhs, rhs)
              : makeAppropriatelyTypedMulOp(b, op->getLoc(), lhs, rhs);
  result->setAttr(kLayoutAttrName, outputLayout);
  setMaterializedAttr(result);
  setAttributeAssociatedWith(result->getResult(0), kLayoutAttrName,
                             outputLayout);
  rewriter.replaceOp(op, result->getResult(0));
  return success();
}

}  // namespace heir
}  // namespace mlir
