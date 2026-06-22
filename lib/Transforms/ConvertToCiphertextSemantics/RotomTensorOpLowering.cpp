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

namespace {

using presburger::IntegerRelation;
using tensor_ext::LayoutAttr;

auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
auto& kMaterializedAttrName = "tensor_ext.layout_materialized";

void setMaterializedAttr(Operation* op) {
  op->setAttr(kMaterializedAttrName, UnitAttr::get(op->getContext()));
}

// Index every (domain -> range) point of a layout's relation by domain tuple in
// a SINGLE enumeration. The matvec kernels verify the diagonal structure of
// every matrix element; doing so with one fixed-domain enumeration per element
// (each of which copies the IntegerRelation) is O(m*n) ISL work and dominates
// convert time at network scale. Enumerating once and indexing turns the
// per-element check into a hash lookup.
using RangeByDomain =
    std::map<std::vector<int64_t>, std::vector<std::pair<int64_t, int64_t>>>;

RangeByDomain indexRangeByDomain(const IntegerRelation& relation) {
  PointPairCollector collector(relation.getNumDomainVars(), /*rangeDims=*/2);
  enumeratePoints(relation, collector);
  RangeByDomain index;
  for (auto& [domain, range] : collector.points) {
    if (range.size() != 2) continue;
    index[domain].push_back({range[0], range[1]});
  }
  return index;
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
RotomTensorOpLowering::getRangePointsForDomain(
    const IntegerRelation& relation, ArrayRef<int64_t> domain) const {
  if (relation.getNumDomainVars() != static_cast<int64_t>(domain.size())) {
    return failure();
  }

  IntegerRelation fixedRelation = fixDomainVars(relation, domain);
  PointCollector collector;
  getRangePoints(fixedRelation, collector);
  if (collector.points.empty()) return failure();
  return collector.points;
}

FailureOr<std::vector<std::vector<int64_t>>>
RotomTensorOpLowering::getRangePointsForDomain(LayoutAttr layout,
                                               ArrayRef<int64_t> domain) const {
  return getRangePointsForDomain(layout.getIntegerRelation(), domain);
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

  // Index each layout's relation once (a single ISL enumeration) rather than
  // rebuilding/copying the IntegerRelation for every (i, j, k) domain point --
  // the same O(m*n) ISL cost the diagonal kernels avoid via indexRangeByDomain.
  RangeByDomain lhsRange = indexRangeByDomain(lhsLayout.getIntegerRelation());
  RangeByDomain rhsRange = indexRangeByDomain(rhsLayout.getIntegerRelation());
  RangeByDomain outRange = indexRangeByDomain(outputLayout.getIntegerRelation());
  // Resolves the single slot a domain point packs into, or nullopt if the
  // packing is replicated/multi-ciphertext (which this kernel cannot express).
  auto uniqueSlot = [](const RangeByDomain& range,
                       std::vector<int64_t> domain) -> std::optional<int64_t> {
    auto it = range.find(domain);
    if (it == range.end() || it->second.size() != 1) return std::nullopt;
    const std::pair<int64_t, int64_t>& point = it->second[0];  // (ct, slot)
    if (point.first != 0) return std::nullopt;
    return point.second;
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
      std::optional<int64_t> dst = uniqueSlot(outRange, {i, j});
      if (!dst) return failure();
      llvm::DenseSet<std::pair<int64_t, int64_t>> shiftsForOutput;
      for (int64_t k = 0; k < n; ++k) {
        std::optional<int64_t> srcLhs = uniqueSlot(lhsRange, {i, k});
        std::optional<int64_t> srcRhs = uniqueSlot(rhsRange, {k, j});
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
  // Pre-build every group's mask (constants independent of the rotations) before
  // emitting any rotate/mul. createMaskForPoints is the only fallible step here,
  // so doing it first means a failure cannot orphan a partially built
  // accumulation chain (the caller would otherwise fall back, leaving dead ops).
  SmallVector<Value> masks;
  masks.reserve(groups.size());
  for (const auto& [shifts, targetPoints] : groups) {
    FailureOr<Value> mask =
        createMaskForPoints(ciphertextSemanticType, targetPoints, b);
    if (failed(mask)) return failure();
    masks.push_back(*mask);
  }

  // Destination-style semantics: the output tensor is the initial accumulator.
  Value acc = output;
  size_t groupIdx = 0;
  for (const auto& [shifts, targetPoints] : groups) {
    (void)targetPoints;
    Value mask = masks[groupIdx++];
    Value rotatedLhs = createRotate(lhs, shifts.first, b);
    Value rotatedRhs = createRotate(rhs, shifts.second, b);
    Operation* product =
        makeAppropriatelyTypedMulOp(b, op.getLoc(), rotatedLhs, rotatedRhs);
    setMaterializedAttr(product);

    Operation* masked = makeAppropriatelyTypedMulOp(
        b, op.getLoc(), product->getResult(0), mask);
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

LogicalResult RotomTensorOpLowering::lowerMatvecCtDiagonalBsgs(
    linalg::MatmulOp op, TypedValue<RankedTensorType> lhs,
    TypedValue<RankedTensorType> rhs, TypedValue<RankedTensorType> output,
    LayoutAttr lhsLayout, LayoutAttr rhsLayout, LayoutAttr outputLayout,
    int64_t m, int64_t n, int64_t p,
    ContextAwareConversionPatternRewriter& rewriter) const {
  // Only a matvec: A(m x n) * x(n x 1) -> out(m x 1).
  if (p != 1) return failure();

  RankedTensorType lhsType = lhs.getType();
  RankedTensorType rhsType = rhs.getType();
  RankedTensorType outType = output.getType();
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outType.getRank() != 2) {
    return failure();
  }
  const int64_t D = lhsType.getDimSize(0);  // diagonals = padded rows = numCt
  const int64_t numSlots = lhsType.getDimSize(1);
  // The diagonal/slot arithmetic runs over the power-of-two padded extents the
  // materializer packed into, not the raw matmul dims (m, n), which need not be
  // powers of two (e.g. an MNIST 128x784 layer). D is the padded diagonal count;
  // Kp is the padded contraction period. The real domain [0, m) x [0, n) is what
  // gets verified -- the [m, D) x [n, Kp) padding is zero (zero-padded matrix and
  // vector) so it contributes nothing to the matvec.
  const int64_t Kp = static_cast<int64_t>(nextPowerOfTwo(n));
  // The matrix is multi-ciphertext (one ciphertext per diagonal); the vector and
  // output occupy a single ciphertext with matching slot count.
  if (D <= 1 || rhsType.getDimSize(0) != 1 || outType.getDimSize(0) != 1 ||
      rhsType.getDimSize(1) != numSlots || outType.getDimSize(1) != numSlots) {
    return failure();
  }
  // D diagonals (>= padded rows), with D | Kp. D == Kp is the square Halevi-Shoup
  // matvec; D < Kp is the squat case (the contraction also spans Kp/D column
  // blocks, collapsed by a residual rotate-and-sum). numSlots is a multiple of
  // Kp: the vector and matrix are replicated period-Kp so cyclic rotations over
  // the full ciphertext wrap mod Kp (numSlots == Kp is the single-period case).
  if (m > D || Kp % D != 0 || numSlots % Kp != 0) return failure();

  // Verify the canonical ciphertext-axis diagonal structure this kernel assumes
  // (each domain point may map to several slots when replicated period-K):
  //   A[i,k] -> ct = (i - k) mod D, slot == k (mod K)
  //   x[k]   -> ct 0, slot == k (mod K)
  //   out[i] -> ct 0, slot i (not replicated)
  // If anything deviates, decline so a safe fallback kernel runs. Index each
  // layout once (one enumeration) and look up every element, instead of one ISL
  // enumeration per element -- the latter dominates convert time at scale.
  const RangeByDomain matrixIndex =
      indexRangeByDomain(lhsLayout.getIntegerRelation());
  const RangeByDomain vectorIndex =
      indexRangeByDomain(rhsLayout.getIntegerRelation());
  const RangeByDomain outputIndex =
      indexRangeByDomain(outputLayout.getIntegerRelation());
  auto matrixOk = [&](int64_t i, int64_t k) -> bool {
    auto it = matrixIndex.find({i, k});
    if (it == matrixIndex.end() || it->second.empty()) return false;
    const int64_t expectedCt = ((i - k) % D + D) % D;
    for (const auto& [ct, slot] : it->second) {
      if (ct != expectedCt || slot % Kp != k) return false;
    }
    return true;
  };
  auto vectorOk = [&](int64_t k) -> bool {
    auto it = vectorIndex.find({k, 0});
    if (it == vectorIndex.end() || it->second.empty()) return false;
    for (const auto& [ct, slot] : it->second) {
      if (ct != 0 || slot % Kp != k) return false;
    }
    return true;
  };
  auto outputOk = [&](int64_t i) -> bool {
    auto it = outputIndex.find({i, 0});
    return it != outputIndex.end() && it->second.size() == 1 &&
           it->second[0].first == 0 && it->second[0].second == i;
  };

  // Verify only the real (unpadded) domain; padding rows/columns are zero.
  for (int64_t k = 0; k < n; ++k)
    if (!vectorOk(k)) return failure();
  for (int64_t i = 0; i < m; ++i)
    if (!outputOk(i)) return failure();
  for (int64_t i = 0; i < m; ++i)
    for (int64_t k = 0; k < n; ++k)
      if (!matrixOk(i, k)) return failure();

  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  // Build the gap mask up front: it is a constant independent of the BSGS
  // accumulation. createMaskForPoints is the only step here that can fail, and
  // for the validated [0, D) output points it will not -- but creating it before
  // any rotate/mul is emitted guarantees a failure cannot orphan a partially
  // built rotation chain (the caller would otherwise fall back, leaving dead
  // ops). gapMask is null in the square single-period case (D == numSlots).
  Value gapMask;
  if (D < numSlots) {
    std::vector<std::vector<int64_t>> outputPoints;
    outputPoints.reserve(D);
    for (int64_t i = 0; i < D; ++i) outputPoints.push_back({0, i});
    FailureOr<Value> mask = createMaskForPoints(outType, outputPoints, b);
    if (failed(mask)) return failure();
    gapMask = *mask;
  }

  auto leftRotateAmount = [&](int64_t byNegative) -> int64_t {
    // A tensor_ext.rotate by s sends slot t to input slot (t + s) mod numSlots;
    // a logical left-rotation by `byNegative` (i.e. rot(v, -byNegative)) is
    // realized by shift = (-byNegative) mod numSlots.
    return ((-byNegative) % numSlots + numSlots) % numSlots;
  };

  // Baby/giant split (mirrors lib/Kernel/KernelImplementation.h) over the D
  // diagonals: baby-step the vector once and reuse those rotations across every
  // giant step.
  const int64_t baby = static_cast<int64_t>(std::ceil(std::sqrt((double)D)));
  const int64_t giant = (D + baby - 1) / baby;

  // Baby steps of the vector: babyX[b] = rotate(x, -b), shared across giants.
  SmallVector<Value> babyX(baby);
  for (int64_t bs = 0; bs < baby; ++bs) {
    babyX[bs] = (bs == 0) ? Value(rhs)
                          : createRotate(rhs, leftRotateAmount(bs), b);
  }

  auto extractDiagonal = [&](int64_t d) -> Value {
    SmallVector<OpFoldResult> offsets{b.getIndexAttr(d), b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), b.getIndexAttr(numSlots)};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1), b.getIndexAttr(1)};
    auto slice =
        tensor::ExtractSliceOp::create(b, rhsType, lhs, offsets, sizes, strides);
    setMaterializedAttr(slice);
    return slice.getResult();
  };

  // BSGS diagonal sum: S = sum_d rotate(diag_d, -d) * rotate(x, -d) over the D
  // diagonals. The matrix rotations fold away for a plaintext matrix; only the
  // ~2*sqrt(D) vector baby/giant rotations are ciphertext rotations.
  Value sum;
  for (int64_t g = 0; g < giant; ++g) {
    Value inner;
    for (int64_t bs = 0; bs < baby; ++bs) {
      const int64_t d = g * baby + bs;
      if (d >= D) break;
      Value diag = extractDiagonal(d);
      Value rotatedDiag =
          (bs == 0) ? diag : createRotate(diag, leftRotateAmount(bs), b);
      Operation* product = makeAppropriatelyTypedMulOp(b, op.getLoc(),
                                                       rotatedDiag, babyX[bs]);
      product->setAttr(kLayoutAttrName, outputLayout);
      setMaterializedAttr(product);
      if (!inner) {
        inner = product->getResult(0);
      } else {
        Operation* add = makeAppropriatelyTypedAddOp(b, op.getLoc(), inner,
                                                     product->getResult(0));
        add->setAttr(kLayoutAttrName, outputLayout);
        setMaterializedAttr(add);
        inner = add->getResult(0);
      }
    }
    if (!inner) continue;
    Value giantRotated =
        (g == 0) ? inner : createRotate(inner, leftRotateAmount(g * baby), b);
    if (!sum) {
      sum = giantRotated;
    } else {
      Operation* add =
          makeAppropriatelyTypedAddOp(b, op.getLoc(), sum, giantRotated);
      add->setAttr(kLayoutAttrName, outputLayout);
      setMaterializedAttr(add);
      sum = add->getResult(0);
    }
  }

  // Squat residual: rotate-and-sum over the Kp/D column blocks (slot stride D).
  // This collapses the blocks into the D-length result and replicates it
  // period-D, so the diagonal rotations above already landed correctly. For the
  // square case (D == Kp) the loop is empty.
  Value resid = sum;
  for (int64_t shift = D; shift < Kp; shift *= 2) {
    Value rotated = createRotate(resid, shift, b);
    Operation* add =
        makeAppropriatelyTypedAddOp(b, op.getLoc(), resid, rotated);
    add->setAttr(kLayoutAttrName, outputLayout);
    setMaterializedAttr(add);
    resid = add->getResult(0);
  }

  // Mask the gap [D, numSlots): the squat residual replicates the D-length
  // result period-D, so zero everything past the valid output rows. The square
  // single-period case (D == numSlots) has no gap and skips this.
  Value result = resid;
  if (gapMask) {
    Operation* masked =
        makeAppropriatelyTypedMulOp(b, op.getLoc(), result, gapMask);
    masked->setAttr(kLayoutAttrName, outputLayout);
    setMaterializedAttr(masked);
    result = masked->getResult(0);
  }

  // Add the destination-style bias.
  Operation* withBias =
      makeAppropriatelyTypedAddOp(b, op.getLoc(), output, result);
  withBias->setAttr(kLayoutAttrName, outputLayout);
  setMaterializedAttr(withBias);
  Value acc = withBias->getResult(0);

  setAttributeAssociatedWith(acc, kLayoutAttrName, outputLayout);
  rewriter.replaceOp(op, acc);
  return success();
}

LogicalResult RotomTensorOpLowering::lowerMatvecDenseDiagonal(
    linalg::MatmulOp op, TypedValue<RankedTensorType> lhs,
    TypedValue<RankedTensorType> rhs, TypedValue<RankedTensorType> output,
    LayoutAttr lhsLayout, LayoutAttr rhsLayout, LayoutAttr outputLayout,
    int64_t m, int64_t n, int64_t p,
    ContextAwareConversionPatternRewriter& rewriter) const {
  if (p != 1) return failure();

  RankedTensorType lhsType = lhs.getType();   // tensor<numCtMat x N>
  RankedTensorType rhsType = rhs.getType();   // tensor<1 x N>
  RankedTensorType outType = output.getType();  // tensor<1 x N>
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outType.getRank() != 2) {
    return failure();
  }
  const int64_t numCtMat = lhsType.getDimSize(0);
  const int64_t N = lhsType.getDimSize(1);  // matrix slot count
  // The diagonals are packed over the padded extents Dp = nextPow2(m) (rows) and
  // Kp = nextPow2(n) (a K-wide slot block per diagonal), not the raw m, n -- a
  // real layer (e.g. MNIST 512x784) need not have power-of-two dims. The real
  // domain [0, m) x [0, n) is verified; the [m, Dp) x [n, Kp) padding is zero.
  const int64_t Dp = static_cast<int64_t>(nextPowerOfTwo(m));
  const int64_t Kp = static_cast<int64_t>(nextPowerOfTwo(n));
  if (Kp <= 0 || N % Kp != 0) return failure();
  const int64_t P = N / Kp;  // diagonals packed per ciphertext
  // Dense only: P > 1 (P == 1 is the single-period kernel). Dp | Kp, and the
  // matrix holds ceil(Dp/P) ciphertexts.
  if (P <= 1 || Dp <= 1 || Kp % Dp != 0) return failure();
  if (numCtMat != (Dp + P - 1) / P) return failure();
  if (rhsType.getDimSize(0) != 1 || outType.getDimSize(0) != 1 ||
      rhsType.getDimSize(1) != N || outType.getDimSize(1) != N) {
    return failure();
  }

  // Verify the dense diagonal structure:
  //   A[i,k] -> ct = floor(((i-k) mod D)/P), slot = ((i-k) mod D mod P)*K + k
  //   x[k]   -> ct 0, slot k (k < K) ;  out[i] -> ct 0, slot i (i < D)
  // Index each layout once (one enumeration) and look up every element, instead
  // of one ISL enumeration per element (which dominates convert time at scale).
  const RangeByDomain matrixIndex =
      indexRangeByDomain(lhsLayout.getIntegerRelation());
  const RangeByDomain vectorIndex =
      indexRangeByDomain(rhsLayout.getIntegerRelation());
  const RangeByDomain outputIndex =
      indexRangeByDomain(outputLayout.getIntegerRelation());
  auto onePoint = [&](const RangeByDomain& index, int64_t a, int64_t b)
      -> std::optional<std::pair<int64_t, int64_t>> {
    auto it = index.find({a, b});
    if (it == index.end() || it->second.size() != 1) return std::nullopt;
    return it->second[0];
  };
  // Verify only the real (unpadded) domain; padding rows/columns are zero.
  for (int64_t k = 0; k < n; ++k) {
    auto xs = onePoint(vectorIndex, k, 0);
    if (!xs || xs->first != 0 || xs->second != k) return failure();
  }
  for (int64_t i = 0; i < m; ++i) {
    auto os = onePoint(outputIndex, i, 0);
    if (!os || os->first != 0 || os->second != i) return failure();
  }
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t k = 0; k < n; ++k) {
      auto mp = onePoint(matrixIndex, i, k);
      if (!mp) return failure();
      const int64_t d = ((i - k) % Dp + Dp) % Dp;
      if (mp->first != d / P || mp->second != (d % P) * Kp + k) return failure();
    }
  }

  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Type elementType = rhsType.getElementType();
  RankedTensorType kType = RankedTensorType::get({1, Kp}, elementType);

  // Build the gap mask up front (a constant independent of the accumulation), so
  // a createMaskForPoints failure -- unreachable for the validated [0, Dp) output
  // points -- cannot orphan a partially built rotation chain. Null when Dp == Kp.
  Value gapMask;
  if (Dp < Kp) {
    std::vector<std::vector<int64_t>> outputPoints;
    outputPoints.reserve(Dp);
    for (int64_t i = 0; i < Dp; ++i) outputPoints.push_back({0, i});
    FailureOr<Value> mask = createMaskForPoints(kType, outputPoints, b);
    if (failed(mask)) return failure();
    gapMask = *mask;
  }

  auto leftRotate = [&](int64_t byNegative) -> int64_t {
    return ((-byNegative) % Kp + Kp) % Kp;
  };
  auto extractBlock = [&](Value src, int64_t ct, int64_t slotOffset) -> Value {
    SmallVector<OpFoldResult> offsets{b.getIndexAttr(ct),
                                      b.getIndexAttr(slotOffset)};
    SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), b.getIndexAttr(Kp)};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1), b.getIndexAttr(1)};
    auto slice = tensor::ExtractSliceOp::create(b, kType, src, offsets, sizes,
                                                strides);
    setMaterializedAttr(slice);
    return slice.getResult();
  };

  // Extract the K-slot vector (slots [0, K) of the single ciphertext).
  Value xK = extractBlock(rhs, 0, 0);

  const int64_t baby = static_cast<int64_t>(std::ceil(std::sqrt((double)Dp)));
  const int64_t giant = (Dp + baby - 1) / baby;
  SmallVector<Value> babyX(baby);
  for (int64_t bs = 0; bs < baby; ++bs) {
    babyX[bs] =
        (bs == 0) ? xK : createRotate(xK, leftRotate(bs), b);
  }

  // BSGS over the Dp diagonals, K-wide; each diagonal is a K-block at its dense
  // (ciphertext, slot-block) location.
  Value sum;
  for (int64_t g = 0; g < giant; ++g) {
    Value inner;
    for (int64_t bs = 0; bs < baby; ++bs) {
      const int64_t d = g * baby + bs;
      if (d >= Dp) break;
      Value diag = extractBlock(lhs, d / P, (d % P) * Kp);
      Value rotatedDiag =
          (bs == 0) ? diag : createRotate(diag, leftRotate(bs), b);
      Operation* product =
          makeAppropriatelyTypedMulOp(b, op.getLoc(), rotatedDiag, babyX[bs]);
      setMaterializedAttr(product);
      if (!inner) {
        inner = product->getResult(0);
      } else {
        Operation* add = makeAppropriatelyTypedAddOp(b, op.getLoc(), inner,
                                                     product->getResult(0));
        setMaterializedAttr(add);
        inner = add->getResult(0);
      }
    }
    if (!inner) continue;
    Value giantRotated =
        (g == 0) ? inner : createRotate(inner, leftRotate(g * baby), b);
    if (!sum) {
      sum = giantRotated;
    } else {
      Operation* add =
          makeAppropriatelyTypedAddOp(b, op.getLoc(), sum, giantRotated);
      setMaterializedAttr(add);
      sum = add->getResult(0);
    }
  }

  // Squat residual rotate-and-sum over the Kp/Dp column blocks (stride Dp).
  Value resid = sum;
  for (int64_t shift = Dp; shift < Kp; shift *= 2) {
    Value rotated = createRotate(resid, shift, b);
    Operation* add = makeAppropriatelyTypedAddOp(b, op.getLoc(), resid, rotated);
    setMaterializedAttr(add);
    resid = add->getResult(0);
  }

  // Mask the gap [Dp, Kp) of the Kp-wide result.
  Value result = resid;
  if (gapMask) {
    Operation* masked =
        makeAppropriatelyTypedMulOp(b, op.getLoc(), result, gapMask);
    setMaterializedAttr(masked);
    result = masked->getResult(0);
  }

  // Place the Kp-wide result into the N-slot output (slots [0, Kp)) and add the
  // destination-style bias.
  auto zero = arith::ConstantOp::create(b, outType, b.getZeroAttr(outType));
  setMaterializedAttr(zero);
  SmallVector<OpFoldResult> offsets{b.getIndexAttr(0), b.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), b.getIndexAttr(Kp)};
  SmallVector<OpFoldResult> strides{b.getIndexAttr(1), b.getIndexAttr(1)};
  auto inserted = tensor::InsertSliceOp::create(b, result, zero.getResult(),
                                                offsets, sizes, strides);
  setMaterializedAttr(inserted);
  Operation* withBias = makeAppropriatelyTypedAddOp(b, op.getLoc(), output,
                                                    inserted.getResult());
  withBias->setAttr(kLayoutAttrName, outputLayout);
  setMaterializedAttr(withBias);
  Value acc = withBias->getResult(0);

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

  LayoutAttr lhsLayout = getLayoutAttr(lhs);
  LayoutAttr rhsLayout = getLayoutAttr(rhs);
  auto outputLayout = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
  if (!lhsLayout || !rhsLayout || !outputLayout) {
    return rewriter.notifyMatchFailure(
        op, "missing tensor_ext.layout attributes for Rotom matmul");
  }

  // Multi-ciphertext diagonal matvec (Halevi-Shoup + baby-step/giant-step). This
  // path uses a different (multi-ciphertext matrix, single-ciphertext vector)
  // shape, so it runs before the matching-shape requirement below.
  if (succeeded(lowerMatvecCtDiagonalBsgs(op, lhs, rhs, output, lhsLayout,
                                          rhsLayout, outputLayout, m, n, p,
                                          rewriter))) {
    return success();
  }
  if (succeeded(lowerMatvecDenseDiagonal(op, lhs, rhs, output, lhsLayout,
                                         rhsLayout, outputLayout, m, n, p,
                                         rewriter))) {
    return success();
  }

  RankedTensorType ciphertextSemanticType = lhs.getType();
  if (rhs.getType() != ciphertextSemanticType ||
      output.getType() != ciphertextSemanticType) {
    return rewriter.notifyMatchFailure(
        op,
        "Rotom matmul currently requires matching ciphertext semantic "
        "tensor shapes");
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
