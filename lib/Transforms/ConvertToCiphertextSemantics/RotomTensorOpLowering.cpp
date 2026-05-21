#include "lib/Transforms/ConvertToCiphertextSemantics/RotomTensorOpLowering.h"

#include <cstdint>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

namespace {

using presburger::IntegerRelation;
using tensor_ext::LayoutAttr;

auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
auto& kMaterializedAttrName = "tensor_ext.layout_materialized";

void setMaterializedAttr(Operation* op) {
  op->setAttr(kMaterializedAttrName, UnitAttr::get(op->getContext()));
}

}  // namespace

LayoutAttr RotomTensorOpLowering::getLayoutAttr(Value value) const {
  auto layoutLookup = typeConverter->getContextualAttr(value);
  if (failed(layoutLookup)) {
    return nullptr;
  }
  return dyn_cast<LayoutAttr>(layoutLookup.value());
}

FailureOr<std::vector<std::vector<int64_t>>>
RotomTensorOpLowering::getRangePointsForDomain(LayoutAttr layout,
                                               ArrayRef<int64_t> domain) const {
  IntegerRelation relation = layout.getIntegerRelation();
  if (relation.getNumDomainVars() != static_cast<int64_t>(domain.size())) {
    return failure();
  }

  IntegerRelation fixedRelation = fixDomainVars(relation, domain);
  PointCollector collector;
  getRangePoints(fixedRelation, collector);
  if (collector.points.empty()) return failure();
  return collector.points;
}

FailureOr<Value> RotomTensorOpLowering::createMaskForPoints(
    RankedTensorType ciphertextSemanticType,
    const std::vector<std::vector<int64_t>>& points,
    ImplicitLocOpBuilder& b) const {
  if (ciphertextSemanticType.getRank() != 2 ||
      !ciphertextSemanticType.hasStaticShape()) {
    return failure();
  }

  Type elementType = ciphertextSemanticType.getElementType();
  SmallVector<Attribute> attrs(ciphertextSemanticType.getNumElements(),
                               b.getZeroAttr(elementType));
  Attribute oneAttr = b.getOneAttr(elementType);
  int64_t slots = ciphertextSemanticType.getDimSize(1);
  for (const std::vector<int64_t>& point : points) {
    if (point.size() != 2) return failure();
    int64_t ct = point[0];
    int64_t slot = point[1];
    if (ct < 0 || slot < 0 || ct >= ciphertextSemanticType.getDimSize(0) ||
        slot >= slots) {
      return failure();
    }
    attrs[ct * slots + slot] = oneAttr;
  }

  auto mask = arith::ConstantOp::create(
      b, ciphertextSemanticType,
      DenseElementsAttr::get(ciphertextSemanticType, attrs));
  setMaterializedAttr(mask);
  return mask.getResult();
}

FailureOr<DenseIntElementsAttr> RotomTensorOpLowering::createRemapAttr(
    MLIRContext* ctx, const std::vector<std::vector<int64_t>>& sourcePoints,
    const std::vector<std::vector<int64_t>>& targetPoints) const {
  SmallVector<int64_t> values;
  for (const std::vector<int64_t>& sourcePoint : sourcePoints) {
    if (sourcePoint.size() != 2) return failure();
    for (const std::vector<int64_t>& targetPoint : targetPoints) {
      if (targetPoint.size() != 2) return failure();
      values.push_back(sourcePoint[0]);
      values.push_back(sourcePoint[1]);
      values.push_back(targetPoint[0]);
      values.push_back(targetPoint[1]);
    }
  }
  if (values.empty()) return failure();

  auto type = RankedTensorType::get(
      {static_cast<int64_t>(values.size() / 4), 4}, IntegerType::get(ctx, 64));
  return DenseIntElementsAttr::get(type, values);
}

FailureOr<Value> RotomTensorOpLowering::alignDomainPointToOutput(
    Value source, RankedTensorType ciphertextSemanticType,
    LayoutAttr sourceLayout, ArrayRef<int64_t> sourceDomain,
    ArrayRef<int64_t> outputDomain, LayoutAttr outputLayout,
    ImplicitLocOpBuilder& b) const {
  FailureOr<std::vector<std::vector<int64_t>>> sourcePoints =
      getRangePointsForDomain(sourceLayout, sourceDomain);
  if (failed(sourcePoints)) return failure();

  FailureOr<std::vector<std::vector<int64_t>>> outputPoints =
      getRangePointsForDomain(outputLayout, outputDomain);
  if (failed(outputPoints)) return failure();

  FailureOr<Value> sourceMask =
      createMaskForPoints(ciphertextSemanticType, *sourcePoints, b);
  if (failed(sourceMask)) return failure();

  Operation* maskedSource =
      makeAppropriatelyTypedMulOp(b, b.getLoc(), source, *sourceMask);
  setMaterializedAttr(maskedSource);

  FailureOr<DenseIntElementsAttr> remapAttr =
      createRemapAttr(b.getContext(), *sourcePoints, *outputPoints);
  if (failed(remapAttr)) return failure();

  auto remap =
      tensor_ext::RemapOp::create(b, maskedSource->getResult(0), *remapAttr);
  setMaterializedAttr(remap);
  setAttributeAssociatedWith(remap.getResult(), kLayoutAttrName, outputLayout);
  return remap.getResult();
}

LogicalResult RotomTensorOpLowering::lowerMatmul(
    linalg::MatmulOp op, linalg::MatmulOp::Adaptor adaptor,
    ContextAwareConversionPatternRewriter& rewriter) const {
  LLVM_DEBUG(
      llvm::dbgs() << "Converting linalg.matmul op with Rotom matmul kernel: "
                   << op << "\n");

  auto lhsType = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto rhsType = cast<RankedTensorType>(op.getInputs()[1].getType());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2 || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !resultType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        op, "Rotom matmul requires static rank-2 tensors");
  }

  int64_t m = lhsType.getDimSize(0);
  int64_t n = lhsType.getDimSize(1);
  int64_t p = rhsType.getDimSize(1);
  if (rhsType.getDimSize(0) != n || resultType.getDimSize(0) != m ||
      resultType.getDimSize(1) != p) {
    return rewriter.notifyMatchFailure(op, "invalid matmul shapes");
  }

  auto lhs = cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[0]);
  auto rhs = cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[1]);
  auto output = cast<TypedValue<RankedTensorType>>(adaptor.getOutputs()[0]);
  RankedTensorType ciphertextSemanticType = lhs.getType();
  if (rhs.getType() != ciphertextSemanticType ||
      output.getType() != ciphertextSemanticType) {
    return rewriter.notifyMatchFailure(
        op,
        "Rotom matmul currently requires matching ciphertext semantic "
        "tensor shapes");
  }

  LayoutAttr lhsLayout = getLayoutAttr(lhs);
  LayoutAttr rhsLayout = getLayoutAttr(rhs);
  auto outputLayout = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
  if (!lhsLayout || !rhsLayout || !outputLayout) {
    return rewriter.notifyMatchFailure(
        op, "missing tensor_ext.layout attributes for Rotom matmul");
  }

  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Value acc = output;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < p; ++j) {
      SmallVector<int64_t> outputDomain = {i, j};
      for (int64_t k = 0; k < n; ++k) {
        FailureOr<Value> alignedLhs =
            alignDomainPointToOutput(lhs, ciphertextSemanticType, lhsLayout,
                                     {i, k}, outputDomain, outputLayout, b);
        if (failed(alignedLhs)) {
          return op.emitError()
                 << "failed to align lhs contribution for Rotom matmul";
        }

        FailureOr<Value> alignedRhs =
            alignDomainPointToOutput(rhs, ciphertextSemanticType, rhsLayout,
                                     {k, j}, outputDomain, outputLayout, b);
        if (failed(alignedRhs)) {
          return op.emitError()
                 << "failed to align rhs contribution for Rotom matmul";
        }

        Operation* product = makeAppropriatelyTypedMulOp(
            b, op.getLoc(), *alignedLhs, *alignedRhs);
        product->setAttr(kLayoutAttrName, outputLayout);
        setMaterializedAttr(product);

        Operation* sum = makeAppropriatelyTypedAddOp(b, op.getLoc(), acc,
                                                     product->getResult(0));
        sum->setAttr(kLayoutAttrName, outputLayout);
        setMaterializedAttr(sum);
        acc = sum->getResult(0);
      }
    }
  }

  setAttributeAssociatedWith(acc, kLayoutAttrName, outputLayout);
  rewriter.replaceOp(op, acc);
  return success();
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
  bool isMul = isa<arith::MulFOp, arith::MulIOp>(op);
  if (!isAdd && !isMul) {
    return rewriter.notifyMatchFailure(op,
                                       "unsupported Rotom elementwise op kind");
  }

  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  auto zero = arith::ConstantOp::create(b, ciphertextSemanticType,
                                        b.getZeroAttr(ciphertextSemanticType));
  zero->setAttr(kLayoutAttrName, outputLayout);
  setMaterializedAttr(zero);
  setAttributeAssociatedWith(zero.getResult(), kLayoutAttrName, outputLayout);

  Value acc = zero.getResult();
  LogicalResult loweringResult = success();
  iterateIndices(
      resultType.getShape(), [&](const std::vector<int64_t>& domain) {
        if (failed(loweringResult)) return;

        FailureOr<Value> alignedLhs = alignDomainPointToOutput(
            adaptorOperands[0], ciphertextSemanticType, lhsLayout, domain,
            domain, outputLayout, b);
        if (failed(alignedLhs)) {
          op->emitError()
              << "failed to align lhs contribution for Rotom elementwise op";
          loweringResult = failure();
          return;
        }

        FailureOr<Value> alignedRhs = alignDomainPointToOutput(
            adaptorOperands[1], ciphertextSemanticType, rhsLayout, domain,
            domain, outputLayout, b);
        if (failed(alignedRhs)) {
          op->emitError()
              << "failed to align rhs contribution for Rotom elementwise op";
          loweringResult = failure();
          return;
        }

        Operation* pointValue =
            isAdd ? makeAppropriatelyTypedAddOp(b, op->getLoc(), *alignedLhs,
                                                *alignedRhs)
                  : makeAppropriatelyTypedMulOp(b, op->getLoc(), *alignedLhs,
                                                *alignedRhs);
        pointValue->setAttr(kLayoutAttrName, outputLayout);
        setMaterializedAttr(pointValue);

        Operation* sum = makeAppropriatelyTypedAddOp(b, op->getLoc(), acc,
                                                     pointValue->getResult(0));
        sum->setAttr(kLayoutAttrName, outputLayout);
        setMaterializedAttr(sum);
        acc = sum->getResult(0);
      });
  if (failed(loweringResult)) return failure();

  setAttributeAssociatedWith(acc, kLayoutAttrName, outputLayout);
  rewriter.replaceOp(op, acc);
  return success();
}

}  // namespace heir
}  // namespace mlir
