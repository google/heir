#include "lib/Transforms/ConvertToCiphertextSemantics/RotomTensorOpLowering.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

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
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project

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
  auto ciphertextSemanticType = dyn_cast<RankedTensorType>(convertedResultType);
  if (!ciphertextSemanticType || lhsType != ciphertextSemanticType ||
      rhsType != ciphertextSemanticType) {
    return rewriter.notifyMatchFailure(
        op,
        "Rotom elementwise lowering currently requires matching ciphertext "
        "semantic tensor shapes");
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

  // Convert-then-compute: bring each operand to the shared output layout with a
  // tensor_ext.convert_layout (a no-op when it already is), then perform the
  // elementwise op once at that layout. The implement-shift-network pass lowers
  // the conversions into rotations + masks.
  auto convertToOutput = [&](Value operand, LayoutAttr operandLayout) -> Value {
    if (operandLayout == outputLayout) return operand;
    Value converted = tensor_ext::ConvertLayoutOp::create(
        b, operand, operandLayout, outputLayout);
    setAttributeAssociatedWith(converted, kLayoutAttrName, outputLayout);
    return converted;
  };
  Value lhs = convertToOutput(adaptorOperands[0], lhsLayout);
  Value rhs = convertToOutput(adaptorOperands[1], rhsLayout);

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
