#include "lib/Transforms/ConvertToCiphertextSemantics/RotomTensorOpLowering.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/MapVector.h"  // from @llvm-project
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

  FailureOr<Value> targetMask =
      createMaskForPoints(ciphertextSemanticType, *outputPoints, b);
  if (failed(targetMask)) return failure();

  Operation* maskedRemap = makeAppropriatelyTypedMulOp(
      b, b.getLoc(), remap.getResult(), *targetMask);
  setMaterializedAttr(maskedRemap);
  setAttributeAssociatedWith(maskedRemap->getResult(0), kLayoutAttrName,
                             outputLayout);
  return maskedRemap->getResult(0);
}

Value RotomTensorOpLowering::createRotate(Value tensor, int64_t shift,
                                          ImplicitLocOpBuilder& b) const {
  auto shiftConst = arith::ConstantOp::create(b, b.getIndexAttr(shift));
  setMaterializedAttr(shiftConst);
  auto rotate = tensor_ext::RotateOp::create(b, tensor, shiftConst);
  setMaterializedAttr(rotate);
  return rotate.getResult();
}

LogicalResult RotomTensorOpLowering::lowerMatmulByRotation(
    linalg::MatmulOp op, Value lhs, Value rhs, Value output,
    RankedTensorType ciphertextSemanticType, LayoutAttr lhsLayout,
    LayoutAttr rhsLayout, LayoutAttr outputLayout, int64_t m, int64_t n,
    int64_t p, ContextAwareConversionPatternRewriter& rewriter) const {
  int64_t numCiphertexts = ciphertextSemanticType.getDimSize(0);
  int64_t numSlots = ciphertextSemanticType.getDimSize(1);
  // The rotation kernel works within a single ciphertext: tensor_ext.rotate
  // shifts each ciphertext's slots independently, so cross-ciphertext movement
  // would need a shift network. Defer those to the brute-force path.
  if (numCiphertexts != 1) return failure();

  // Resolves the single slot a domain point packs into, or nullopt if the
  // packing is replicated/multi-ciphertext (which this kernel cannot express).
  auto uniqueSlot = [&](LayoutAttr layout,
                        ArrayRef<int64_t> domain) -> std::optional<int64_t> {
    FailureOr<std::vector<std::vector<int64_t>>> points =
        getRangePointsForDomain(layout, domain);
    if (failed(points) || points->size() != 1) return std::nullopt;
    const std::vector<int64_t>& point = (*points)[0];
    if (point.size() != 2 || point[0] != 0) return std::nullopt;
    return point[1];
  };

  // Group every (i, j, k) contribution by the (lhsShift, rhsShift) pair that
  // realizes it. A left-rotation by s maps output slot dst to input slot
  // (dst + s) mod numSlots, so placing input slot `src` at `dst` needs
  // s = (src - dst) mod numSlots. Each group records the output slots it
  // writes, which become its mask.
  llvm::MapVector<std::pair<int64_t, int64_t>, std::vector<std::vector<int64_t>>>
      groups;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < p; ++j) {
      std::optional<int64_t> dst = uniqueSlot(outputLayout, {i, j});
      if (!dst) return failure();
      llvm::DenseSet<std::pair<int64_t, int64_t>> shiftsForOutput;
      for (int64_t k = 0; k < n; ++k) {
        std::optional<int64_t> srcLhs = uniqueSlot(lhsLayout, {i, k});
        std::optional<int64_t> srcRhs = uniqueSlot(rhsLayout, {k, j});
        if (!srcLhs || !srcRhs) return failure();
        int64_t lhsShift = ((*srcLhs - *dst) % numSlots + numSlots) % numSlots;
        int64_t rhsShift = ((*srcRhs - *dst) % numSlots + numSlots) % numSlots;
        std::pair<int64_t, int64_t> key = {lhsShift, rhsShift};
        // Two contraction terms for one output slot needing identical shifts
        // would collapse into a single product; bail to the safe path. (This
        // cannot occur for an injective diagonal packing.)
        if (!shiftsForOutput.insert(key).second) return failure();
        groups[key].push_back({0, *dst});
      }
    }
  }
  if (groups.empty()) return failure();

  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  // Destination-style semantics: the output tensor is the initial accumulator.
  Value acc = output;
  for (const auto& [shifts, targetPoints] : groups) {
    Value rotatedLhs = createRotate(lhs, shifts.first, b);
    Value rotatedRhs = createRotate(rhs, shifts.second, b);
    Operation* product =
        makeAppropriatelyTypedMulOp(b, op.getLoc(), rotatedLhs, rotatedRhs);
    setMaterializedAttr(product);

    FailureOr<Value> mask =
        createMaskForPoints(ciphertextSemanticType, targetPoints, b);
    if (failed(mask)) return failure();
    Operation* masked =
        makeAppropriatelyTypedMulOp(b, op.getLoc(), product->getResult(0),
                                    *mask);
    masked->setAttr(kLayoutAttrName, outputLayout);
    setMaterializedAttr(masked);

    Operation* sum = makeAppropriatelyTypedAddOp(b, op.getLoc(), acc,
                                                 masked->getResult(0));
    sum->setAttr(kLayoutAttrName, outputLayout);
    setMaterializedAttr(sum);
    acc = sum->getResult(0);
  }

  setAttributeAssociatedWith(acc, kLayoutAttrName, outputLayout);
  rewriter.replaceOp(op, acc);
  return success();
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

  // Prefer the rotate-multiply-accumulate kernel when the layouts admit it; it
  // collapses each contraction into a small set of ciphertext rotations, which
  // is the payoff of rolled (diagonal) layouts. Falls back to the brute-force
  // per-scalar lowering below when not applicable.
  if (succeeded(lowerMatmulByRotation(op, lhs, rhs, output,
                                      ciphertextSemanticType, lhsLayout,
                                      rhsLayout, outputLayout, m, n, p,
                                      rewriter))) {
    return success();
  }

  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  // Preserve linalg.matmul destination-style semantics: the output tensor is
  // the initial accumulator, and each scalar product is added into it.
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
