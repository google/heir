#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"     // from @llvm-project

namespace mlir::heir::rotom {

namespace {

constexpr llvm::StringLiteral kRotomSeedAttrName = "rotom.seed";
constexpr llvm::StringLiteral kRotomLayoutAttrName = "rotom.layout";

}  // namespace

#define DEBUG_TYPE "rotom-assign-layout"

#define GEN_PASS_DEF_LAYOUTASSIGNMENT
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h.inc"

namespace {

enum class KernelKind {
  Tensor,
  BlockArgument,
  Yield,
  PassThrough,
  Elementwise,
  Generic,
  Matmul,
  Transpose,
  Reduce,
  CollapseShape,
  ExpandShape,
  ExtractSlice,
  InsertSlice,
};

llvm::StringLiteral kernelKindName(KernelKind kind) {
  switch (kind) {
    case KernelKind::Tensor:
      return "tensor";
    case KernelKind::BlockArgument:
      return "block_arg";
    case KernelKind::Yield:
      return "yield";
    case KernelKind::PassThrough:
      return "pass_through";
    case KernelKind::Elementwise:
      return "elementwise";
    case KernelKind::Generic:
      return "generic";
    case KernelKind::Matmul:
      return "matmul";
    case KernelKind::Transpose:
      return "transpose";
    case KernelKind::Reduce:
      return "reduce";
    case KernelKind::CollapseShape:
      return "collapse_shape";
    case KernelKind::ExpandShape:
      return "expand_shape";
    case KernelKind::ExtractSlice:
      return "extract_slice";
    case KernelKind::InsertSlice:
      return "insert_slice";
  }
  llvm_unreachable("unknown kernel kind");
}

struct Candidate {
  LayoutAttr layout;
  int64_t cost = 0;
  KernelKind kind = KernelKind::PassThrough;
  SmallVector<Value> operands;
  SmallVector<LayoutAttr> operandLayouts;
  std::optional<KernelName> kernel;
};

std::string layoutKey(LayoutAttr layout) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << layout;
  return storage;
}

std::string kernelKey(std::optional<KernelName> kernel) {
  if (!kernel) return "none";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << *kernel;
  return storage;
}

SmallVector<Candidate> uniqueCandidates(ArrayRef<Candidate> candidates);

Type getPlainValueType(Type type) {
  if (auto secretType = dyn_cast<secret::SecretType>(type)) {
    return secretType.getValueType();
  }
  return type;
}

bool isTensorLike(Value value) {
  return isa<RankedTensorType>(getPlainValueType(value.getType()));
}

bool isLayoutCompatibleWithValue(LayoutAttr layout, Value value) {
  auto type = dyn_cast<RankedTensorType>(getPlainValueType(value.getType()));
  if (!type) return false;

  int64_t rank = type.getRank();
  for (Attribute attr : layout.getDims()) {
    auto dim = cast<DimAttr>(attr);
    if (dim.isGap() || dim.isReplicate()) continue;
    int64_t dimIndex = dim.getDim();
    if (dimIndex >= rank) return false;
    int64_t typeDimSize = type.getDimSize(dimIndex);
    if (typeDimSize == ShapedType::kDynamic) continue;
    if (typeDimSize <= 0) continue;
    int64_t paddedDimSize = nextPowerOfTwo(typeDimSize);
    if (dim.getSize() * dim.getStride() > paddedDimSize) return false;
  }
  return true;
}

bool hasOnlyUnitStrides(ArrayRef<int64_t> strides) {
  return llvm::all_of(strides, [](int64_t stride) { return stride == 1; });
}

bool isDynamic(int64_t value) { return value == ShapedType::kDynamic; }

int64_t layoutConversionCost(LayoutAttr from, LayoutAttr to) {
  if (from == to) return 0;
  return 4 + std::abs(layoutNumCiphertexts(from) - layoutNumCiphertexts(to));
}

bool isAddLike(Operation* op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(op);
}

bool isAdd(Operation* op) { return isa<arith::AddFOp, arith::AddIOp>(op); }

bool isMulLike(Operation* op) { return isa<arith::MulFOp, arith::MulIOp>(op); }

std::optional<KernelName> selectRotomElementwiseKernel(Operation* op) {
  if (isAdd(op)) return KernelName::RotomAdd;
  if (isMulLike(op)) return KernelName::RotomMul;
  return std::nullopt;
}

int64_t operationCost(Operation* op, LayoutAttr layout) {
  int64_t numCt = layoutNumCiphertexts(layout);
  if (isAddLike(op)) return numCt;
  if (isMulLike(op)) return 10 * numCt;
  return 0;
}

// Cost penalty for a matmul candidate that has no selectable kernel and thus
// cannot be lowered; large enough to dominate any real rotation cost so a
// kernel-bearing candidate is always preferred.
constexpr int64_t kUnloweredMatmulPenalty = 1LL << 50;

int64_t ceilSqrt(int64_t value) {
  if (value <= 1) return 1;
  int64_t root = 1;
  while (root * root < value) ++root;
  return root;
}

int64_t ceilLog2(int64_t value) {
  if (value <= 1) return 0;
  int64_t bits = 0;
  int64_t pow = 1;
  while (pow < value) {
    pow *= 2;
    ++bits;
  }
  return bits;
}

// Rotation-aware matmul cost so the search can prefer a diagonal (rolled,
// ciphertext-axis) packing over a row-major one. Mirrors the reference Rotom
// cost model (ir/kernel_cost.py matmul_ops / bsgs_matmul_ops): a ciphertext-axis
// diagonal matvec lowers with a baby-step/giant-step schedule -- ~2*sqrt(K)
// ciphertext-vector rotations plus per-diagonal matrix rotations that fold to
// free for a plaintext matrix -- whereas the row-major rotate-multiply-
// accumulate kernel emits O(m*K*p) rotations.
int64_t matmulRotationCost(LayoutAttr lhsLayout, RankedTensorType lhsType,
                           RankedTensorType rhsType) {
  constexpr int64_t kRotWeight = 4;
  int64_t m = nextPowerOfTwo(lhsType.getDimSize(0));
  int64_t k = nextPowerOfTwo(lhsType.getDimSize(1));
  int64_t p = nextPowerOfTwo(rhsType.getDimSize(1));
  DenseI64ArrayAttr rolls = lhsLayout.getRolls();
  // A rolled lhs is the diagonal kernel regardless of ciphertext count: it can be
  // sparse (one diagonal/ct), dense (P diagonals/ct), or single-ciphertext
  // (numCt == 1 when the whole matrix fits one ciphertext).
  bool diagonal = rolls && !rolls.empty();
  if (diagonal && p == 1 && m <= k && k % m == 0) {
    // Diagonal Halevi-Shoup matvec: baby-step/giant-step over the M (output-row)
    // diagonals plus a log-depth residual rotate-and-sum over the K/M column
    // blocks (empty when M == K). ~2*sqrt(M) ciphertext-vector rotations + the
    // residual; the M matrix-diagonal rotations fold to free for a plaintext
    // matrix. Counted in rotations (the dominant ciphertext cost) so it is
    // comparable to the row-major rotate-multiply-accumulate path below.
    int64_t rotations = 2 * ceilSqrt(m) + m + ceilLog2(k / m);
    return kRotWeight * rotations;
  }
  return kRotWeight * 2 * m * k * p;
}

// Builds the ciphertext-axis diagonal layout for an M x K matrix: with n == K
// the derived ct/slot split puts the column dim in slots and the row dim on the
// ciphertext axis, and roll(0,1) makes ct = (row - col) mod M -- one ciphertext
// per diagonal. M == K is the square diagonal; M < K (M | K) is the squat
// diagonal. This is exactly the packing the diagonal matvec kernels verify.
LayoutAttr makeCtDiagonalMatrixLayout(MLIRContext* ctx, int64_t m, int64_t k,
                                      int64_t n) {
  SmallVector<Attribute> dims = {DimAttr::get(ctx, /*dim=*/0, m, /*stride=*/1),
                                 DimAttr::get(ctx, /*dim=*/1, k, /*stride=*/1)};
  LayoutAttr base = LayoutAttr::get(ctx, ArrayAttr::get(ctx, dims), n);
  SmallVector<std::pair<int64_t, int64_t>> rolls = {{0, 1}};
  return withRolls(base, rolls);
}

// Identity-packed vector/output layout (slot = index, single ciphertext) -- the
// rhs the diagonal matvec kernel rotates, or the M-length output. The trailing
// singleton keeps the layout's domain rank 2 to match the K x 1 / M x 1 tensor.
LayoutAttr makeIdentityVectorLayout(MLIRContext* ctx, int64_t k, int64_t n) {
  SmallVector<Attribute> dims = {DimAttr::get(ctx, /*dim=*/0, k, /*stride=*/1),
                                 DimAttr::get(ctx, /*dim=*/1, 1, /*stride=*/1)};
  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, dims), n);
}

int64_t genericOperationCost(linalg::GenericOp op, LayoutAttr layout) {
  int64_t cost = 0;
  for (Operation& innerOp : op.getBody()->getOperations()) {
    if (isa<linalg::YieldOp, arith::ConstantOp>(innerOp)) continue;
    cost += operationCost(&innerOp, layout);
  }
  return cost;
}

std::string candidateTieKey(const Candidate& candidate) {
  std::string key = kernelKindName(candidate.kind).str();
  key += ":";
  key += layoutKey(candidate.layout);
  key += ":kernel=";
  key += kernelKey(candidate.kernel);
  for (LayoutAttr operandLayout : candidate.operandLayouts) {
    key += ":";
    key += layoutKey(operandLayout);
  }
  return key;
}

bool isBetterCandidate(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.cost != rhs.cost) return lhs.cost < rhs.cost;
  if (lhs.kernel.has_value() != rhs.kernel.has_value()) {
    return lhs.kernel.has_value();
  }
  return candidateTieKey(lhs) < candidateTieKey(rhs);
}

std::optional<LayoutAttr> remapLayoutDims(LayoutAttr layout,
                                          ArrayRef<int64_t> oldToNewDim) {
  SmallVector<Attribute> dims;
  MLIRContext* ctx = layout.getContext();
  for (Attribute attr : layout.getDims()) {
    auto dim = cast<DimAttr>(attr);
    if (dim.isGap() || dim.isReplicate()) {
      dims.push_back(dim);
      continue;
    }

    int64_t oldDim = dim.getDim();
    if (oldDim < 0 || oldDim >= static_cast<int64_t>(oldToNewDim.size())) {
      return std::nullopt;
    }

    int64_t newDim = oldToNewDim[oldDim];
    if (newDim == -1) continue;
    if (newDim < -1) return std::nullopt;
    dims.push_back(DimAttr::get(ctx, newDim, dim.getSize(), dim.getStride()));
  }

  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, dims), layout.getN());
}

std::optional<LayoutAttr> combineMatmulOutputLayout(LayoutAttr lhsLayout,
                                                    LayoutAttr rhsLayout) {
  if (lhsLayout.getN() != rhsLayout.getN()) return std::nullopt;

  MLIRContext* ctx = lhsLayout.getContext();
  SmallVector<Attribute> dims;
  for (Attribute attr : lhsLayout.getDims()) {
    auto dim = cast<DimAttr>(attr);
    if (dim.isGap() || dim.isReplicate()) {
      dims.push_back(dim);
      continue;
    }
    if (dim.getDim() == 0) {
      dims.push_back(
          DimAttr::get(ctx, /*dim=*/0, dim.getSize(), dim.getStride()));
    }
  }
  for (Attribute attr : rhsLayout.getDims()) {
    auto dim = cast<DimAttr>(attr);
    if (dim.isGap() || dim.isReplicate()) {
      dims.push_back(dim);
      continue;
    }
    if (dim.getDim() == 1) {
      dims.push_back(
          DimAttr::get(ctx, /*dim=*/1, dim.getSize(), dim.getStride()));
    }
  }

  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, dims), lhsLayout.getN());
}

bool hasStaticPositiveShape(RankedTensorType type) {
  return llvm::all_of(type.getShape(), [](int64_t dim) {
    return dim != ShapedType::kDynamic && dim > 0;
  });
}

std::optional<presburger::IntegerRelation> lowerRotomLayoutToRelation(
    LayoutAttr layout) {
  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  if (failed(isl)) return std::nullopt;

  FailureOr<presburger::IntegerRelation> relation =
      getIntegerRelationFromIslStr(*isl);
  if (failed(relation)) return std::nullopt;
  return *relation;
}

bool isBicyclicLayoutFor(LayoutAttr layout, RankedTensorType type) {
  if (type.getRank() != 2 || !hasStaticPositiveShape(type)) return false;
  if (std::gcd(type.getDimSize(0), type.getDimSize(1)) != 1) return false;

  std::optional<presburger::IntegerRelation> relation =
      lowerRotomLayoutToRelation(layout);
  if (!relation) return false;

  return isRelationBicyclic(type, layout.getN(), *relation);
}

std::optional<KernelName> selectMatmulKernel(RankedTensorType lhsType,
                                             RankedTensorType rhsType,
                                             RankedTensorType resultType,
                                             LayoutAttr lhsLayout,
                                             LayoutAttr rhsLayout,
                                             LayoutAttr resultLayout) {
  if (!hasStaticPositiveShape(lhsType) || !hasStaticPositiveShape(rhsType) ||
      !hasStaticPositiveShape(resultType)) {
    return std::nullopt;
  }

  int64_t m = lhsType.getDimSize(0);
  int64_t n = lhsType.getDimSize(1);
  int64_t rhsN = rhsType.getDimSize(0);
  int64_t p = rhsType.getDimSize(1);
  if (rhsN != n || resultType.getDimSize(0) != m ||
      resultType.getDimSize(1) != p) {
    return std::nullopt;
  }

  if (lhsLayout.getN() != rhsLayout.getN() ||
      lhsLayout.getN() != resultLayout.getN()) {
    return std::nullopt;
  }

  // Prefer the specialized Bicyclic matmul kernel when the layouts satisfy its
  // real packing contract.
  if (std::gcd(m, n) == 1 && std::gcd(n, p) == 1 && std::gcd(m, p) == 1 &&
      isBicyclicLayoutFor(lhsLayout, lhsType) &&
      isBicyclicLayoutFor(rhsLayout, rhsType) &&
      isBicyclicLayoutFor(resultLayout, resultType)) {
    return KernelName::MatmulBicyclic;
  }

  if (supportsRotomAlignmentLowering(lhsLayout, rhsLayout, resultLayout)) {
    return KernelName::RotomMatmul;
  }

  // Ciphertext-axis diagonal matvec (p == 1): the matrix is packed as diagonals
  // (rolled) -- one ciphertext per diagonal (sparse, numCiphertexts == m), or
  // P = N/K diagonals per ciphertext (dense, numCiphertexts == ceil(m/P)) --
  // against a single-ciphertext vector and result. m == n is the square baby-
  // step/giant-step kernel, m < n (n % m == 0) the squat kernel. The precise
  // diagonal structure (including the ciphertext count) is verified in
  // RotomTensorOpLowering, so this gate stays loose and the kernel declines
  // (falling back) if it does not match.
  DenseI64ArrayAttr lhsRolls = lhsLayout.getRolls();
  // The diagonal layout pads M and K to powers of two (the slot/ciphertext
  // extents the kernel actually walks), so the square/squat divisibility
  // contract must be checked on the padded extents -- not the raw type dims,
  // which need not be powers of two (e.g. an MNIST 128x784 or 10x128 matmul).
  int64_t mp = nextPowerOfTwo(m);
  int64_t np = nextPowerOfTwo(n);
  if (p == 1 && lhsRolls && !lhsRolls.empty() && mp <= np && np % mp == 0 &&
      layoutNumCiphertexts(rhsLayout) == 1 &&
      layoutNumCiphertexts(resultLayout) == 1) {
    return KernelName::RotomMatmul;
  }

  return std::nullopt;
}

SmallVector<Candidate> remapCandidates(Value operand,
                                       ArrayRef<Candidate> candidates,
                                       ArrayRef<int64_t> oldToNewDim,
                                       KernelKind kind, int64_t extraCost = 0) {
  SmallVector<Candidate> remapped;
  for (const Candidate& candidate : candidates) {
    std::optional<LayoutAttr> layout =
        remapLayoutDims(candidate.layout, oldToNewDim);
    if (!layout) continue;
    remapped.push_back({*layout,
                        candidate.cost + extraCost,
                        kind,
                        {operand},
                        {candidate.layout}});
  }
  return uniqueCandidates(remapped);
}

SmallVector<Candidate> chooseCommonCandidates(
    ArrayRef<Value> operands, ArrayRef<SmallVector<Candidate>> candidateSets,
    KernelKind kind, function_ref<int64_t(LayoutAttr)> localCostFn) {
  if (operands.size() != candidateSets.size()) return {};

  SmallVector<Candidate> targets;
  for (const SmallVector<Candidate>& candidates : candidateSets) {
    for (const Candidate& candidate : candidates) {
      targets.push_back({candidate.layout, 0, kind});
    }
  }
  targets = uniqueCandidates(targets);
  if (targets.empty()) return {};

  SmallVector<Candidate> chosen;
  for (const Candidate& target : targets) {
    int64_t totalCost = localCostFn(target.layout);
    bool valid = true;
    SmallVector<LayoutAttr> operandLayouts;
    for (const SmallVector<Candidate>& candidates : candidateSets) {
      if (candidates.empty()) continue;
      const Candidate* bestCandidate = nullptr;
      std::optional<Candidate> bestScoredCandidate;
      int64_t bestCost = 0;
      for (const Candidate& candidate : candidates) {
        int64_t conversionCost =
            layoutConversionCost(candidate.layout, target.layout);
        int64_t cost = candidate.cost + conversionCost;
        Candidate scoredCandidate = candidate;
        scoredCandidate.cost = cost;
        scoredCandidate.kind = kind;
        if (!bestScoredCandidate ||
            isBetterCandidate(scoredCandidate, *bestScoredCandidate)) {
          bestScoredCandidate = scoredCandidate;
          bestCandidate = &candidate;
          bestCost = cost;
        }
      }
      if (!bestCandidate) {
        valid = false;
        break;
      }
      totalCost += bestCost;
      operandLayouts.push_back(bestCandidate->layout);
    }
    if (valid) {
      chosen.push_back({target.layout, totalCost, kind,
                        SmallVector<Value>(operands), operandLayouts});
    }
  }
  return uniqueCandidates(chosen);
}

std::optional<SmallVector<int64_t>> getReductionDimMap(
    int64_t inputRank, ArrayRef<int64_t> reductionDims) {
  SmallVector<bool> isReduced(inputRank, false);
  for (int64_t dim : reductionDims) {
    if (dim < 0 || dim >= inputRank) return std::nullopt;
    isReduced[dim] = true;
  }

  SmallVector<int64_t> oldToNew(inputRank, -1);
  int64_t newDim = 0;
  for (int64_t dim = 0; dim < inputRank; ++dim) {
    if (isReduced[dim]) continue;
    oldToNew[dim] = newDim++;
  }
  return oldToNew;
}

std::optional<SmallVector<int64_t>> getCollapseShapeDimMap(
    RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  SmallVector<int64_t> oldToNew(sourceType.getRank(), -2);

  for (auto [resultDim, group] : llvm::enumerate(reassociationIndices)) {
    int64_t mappedDim = -1;
    for (int64_t sourceDim : group) {
      if (sourceDim < 0 || sourceDim >= sourceType.getRank()) {
        return std::nullopt;
      }

      int64_t dimSize = sourceType.getDimSize(sourceDim);
      if (dimSize == 1) {
        oldToNew[sourceDim] = -1;
        if (mappedDim == -1) mappedDim = sourceDim;
        continue;
      }
      if (isDynamic(dimSize)) return std::nullopt;
      if (mappedDim != -1 && sourceType.getDimSize(mappedDim) != 1) {
        return std::nullopt;
      }
      mappedDim = sourceDim;
    }

    if (mappedDim == -1) return std::nullopt;
    oldToNew[mappedDim] = static_cast<int64_t>(resultDim);
  }

  return oldToNew;
}

std::optional<SmallVector<int64_t>> getExpandShapeDimMap(
    RankedTensorType resultType,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  SmallVector<int64_t> oldToNew;
  oldToNew.reserve(reassociationIndices.size());

  for (ArrayRef<int64_t> group : reassociationIndices) {
    int64_t mappedDim = -1;
    for (int64_t resultDim : group) {
      if (resultDim < 0 || resultDim >= resultType.getRank()) {
        return std::nullopt;
      }

      int64_t dimSize = resultType.getDimSize(resultDim);
      if (dimSize == 1) {
        if (mappedDim == -1) mappedDim = resultDim;
        continue;
      }
      if (isDynamic(dimSize)) return std::nullopt;
      if (mappedDim != -1 && resultType.getDimSize(mappedDim) != 1) {
        return std::nullopt;
      }
      mappedDim = resultDim;
    }
    if (mappedDim == -1) return std::nullopt;
    oldToNew.push_back(mappedDim);
  }

  return oldToNew;
}

std::optional<SmallVector<int64_t>> getExtractSliceDimMap(
    RankedTensorType resultType, ArrayRef<int64_t> staticSizes,
    ArrayRef<int64_t> staticStrides) {
  if (!hasOnlyUnitStrides(staticStrides)) return std::nullopt;

  int64_t sourceRank = static_cast<int64_t>(staticSizes.size());
  int64_t resultRank = resultType.getRank();
  if (sourceRank == resultRank) {
    SmallVector<int64_t> identity(sourceRank);
    std::iota(identity.begin(), identity.end(), 0);
    return identity;
  }

  SmallVector<int64_t> oldToNew(sourceRank, -2);
  int64_t resultDim = 0;
  for (int64_t sourceDim = 0; sourceDim < sourceRank; ++sourceDim) {
    int64_t size = staticSizes[sourceDim];
    if (isDynamic(size)) return std::nullopt;

    if (resultDim < resultRank && size == resultType.getDimSize(resultDim)) {
      oldToNew[sourceDim] = resultDim++;
      continue;
    }
    if (size == 1) {
      oldToNew[sourceDim] = -1;
      continue;
    }
    return std::nullopt;
  }

  if (resultDim != resultRank) return std::nullopt;
  return oldToNew;
}

std::optional<SmallVector<int64_t>> getInsertSliceDimMap(
    RankedTensorType sourceType, RankedTensorType resultType,
    ArrayRef<int64_t> staticSizes, ArrayRef<int64_t> staticStrides) {
  if (!hasOnlyUnitStrides(staticStrides)) return std::nullopt;

  int64_t sourceRank = sourceType.getRank();
  int64_t resultRank = resultType.getRank();
  if (sourceRank == resultRank) {
    SmallVector<int64_t> identity(sourceRank);
    std::iota(identity.begin(), identity.end(), 0);
    return identity;
  }

  SmallVector<int64_t> sourceToResult(sourceRank, -2);
  int64_t sourceDim = 0;
  for (int64_t resultDim = 0; resultDim < resultRank; ++resultDim) {
    int64_t size = staticSizes[resultDim];
    if (isDynamic(size)) return std::nullopt;

    if (sourceDim < sourceRank && size == sourceType.getDimSize(sourceDim)) {
      sourceToResult[sourceDim++] = resultDim;
      continue;
    }
    if (size == 1) continue;
    return std::nullopt;
  }

  if (sourceDim != sourceRank) return std::nullopt;
  return sourceToResult;
}

bool isElementwiseGeneric(linalg::GenericOp op) {
  for (AffineMap map : op.getIndexingMapsArray()) {
    if (!map.isIdentity()) return false;
  }
  for (utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) return false;
  }
  return true;
}

bool hasAddLikeBody(linalg::GenericOp op) {
  bool foundAddLikeOp = false;
  for (Operation& innerOp : op.getBody()->getOperations()) {
    if (isa<linalg::YieldOp, arith::ConstantOp>(innerOp)) continue;
    if (!isAddLike(&innerOp)) return false;
    foundAddLikeOp = true;
  }
  return foundAddLikeOp;
}

SmallVector<Candidate> uniqueCandidates(ArrayRef<Candidate> candidates) {
  SmallVector<Candidate> result;
  for (const Candidate& candidate : candidates) {
    auto it = llvm::find_if(result, [&](const Candidate& existing) {
      return existing.layout == candidate.layout &&
             existing.kernel == candidate.kernel;
    });
    if (it == result.end()) {
      result.push_back(candidate);
      continue;
    }
    if (isBetterCandidate(candidate, *it)) {
      *it = candidate;
    }
  }
  llvm::sort(result, [](const Candidate& lhs, const Candidate& rhs) {
    return isBetterCandidate(lhs, rhs);
  });
  return result;
}

struct LayoutAssignment : public impl::LayoutAssignmentBase<LayoutAssignment> {
  using LayoutAssignmentBase::LayoutAssignmentBase;

  DenseMap<Value, SmallVector<Candidate>> candidates;
  DenseMap<Value, LayoutAttr> selectedLayouts;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<secret::SecretDialect>();
  }

  void seedValue(Value value);
  void setCandidates(Value value, ArrayRef<Candidate> newCandidates);
  std::optional<Candidate> bestCandidate(Value value);
  const Candidate* findCandidate(Value value, LayoutAttr layout);
  void applySelectedKernel(Value value, const Candidate& candidate);
  void markSelected(Value value, LayoutAttr layout);
  SmallVector<Candidate> candidatesForValue(Value value);
  SmallVector<Candidate> chooseCommonOperandCandidates(Operation* op);
  SmallVector<Candidate> chooseCommonOperandCandidates(Operation* op,
                                                       KernelKind kind);
  SmallVector<Candidate> chooseAlignedElementwiseCandidates(
      ArrayRef<Value> operands, KernelKind kind,
      function_ref<int64_t(LayoutAttr)> localCostFn,
      std::optional<KernelName> rotomKernel = std::nullopt);
  void assignResultsFromCandidates(Operation* op, ArrayRef<Candidate> chosen);
  LogicalResult visitOperation(Operation* op);
  LogicalResult visitFunc(func::FuncOp op);
  LogicalResult visitReturn(func::ReturnOp op);
  LogicalResult visitGeneric(secret::GenericOp op);
  LogicalResult visitYield(secret::YieldOp op);
  LogicalResult visitPassThrough(Operation* op);
  LogicalResult visitElementwise(Operation* op);
  LogicalResult visitGeneric(linalg::GenericOp op);
  LogicalResult visitMatmul(linalg::MatmulOp op);
  LogicalResult visitTranspose(linalg::TransposeOp op);
  LogicalResult visitReduction(linalg::ReduceOp op);
  LogicalResult visitCollapseShape(tensor::CollapseShapeOp op);
  LogicalResult visitExpandShape(tensor::ExpandShapeOp op);
  LogicalResult visitExtractSlice(tensor::ExtractSliceOp op);
  LogicalResult visitInsertSlice(tensor::InsertSliceOp op);
  void writeSelectedLayouts();

  void runOnOperation() override;
};

void LayoutAssignment::seedValue(Value value) {
  if (candidates.contains(value)) return;

  FailureOr<Attribute> seedAttr =
      findAttributeAssociatedWith(value, kRotomSeedAttrName);
  if (failed(seedAttr)) return;

  auto seed = dyn_cast<SeedAttr>(*seedAttr);
  if (!seed) return;

  SmallVector<Candidate> seeded;
  for (Attribute attr : seed.getLayouts()) {
    auto layout = dyn_cast<LayoutAttr>(attr);
    if (!layout) continue;
    seeded.push_back({layout, 0, KernelKind::Tensor});
  }
  if (!seeded.empty()) setCandidates(value, seeded);
}

void LayoutAssignment::setCandidates(Value value,
                                     ArrayRef<Candidate> newCandidates) {
  if (!isTensorLike(value) || newCandidates.empty()) return;
  SmallVector<Candidate> compatibleCandidates;
  for (const Candidate& candidate : newCandidates) {
    if (isLayoutCompatibleWithValue(candidate.layout, value)) {
      compatibleCandidates.push_back(candidate);
    }
  }
  if (compatibleCandidates.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No layout candidate is compatible with value "
                            << value << "\n");
    return;
  }

  candidates[value] = uniqueCandidates(compatibleCandidates);
  LLVM_DEBUG({
    llvm::dbgs() << "Assigned " << candidates[value].size()
                 << " candidate(s) to value " << value << "\n";
  });
}

std::optional<Candidate> LayoutAssignment::bestCandidate(Value value) {
  seedValue(value);
  auto it = candidates.find(value);
  if (it == candidates.end() || it->second.empty()) return std::nullopt;
  return it->second.front();
}

const Candidate* LayoutAssignment::findCandidate(Value value,
                                                 LayoutAttr layout) {
  seedValue(value);
  auto it = candidates.find(value);
  if (it == candidates.end()) return nullptr;
  auto candidateIt = llvm::find_if(it->second, [&](const Candidate& candidate) {
    return candidate.layout == layout;
  });
  if (candidateIt == it->second.end()) return nullptr;
  return &*candidateIt;
}

void LayoutAssignment::applySelectedKernel(Value value,
                                           const Candidate& candidate) {
  Operation* op = value.getDefiningOp();
  if (!op) return;

  auto existingKernel = op->getAttrOfType<secret::KernelAttr>(
      secret::SecretDialect::kKernelAttrName);
  if (existingKernel && existingKernel.getForce()) return;

  if (!candidate.kernel) {
    if (candidate.kind == KernelKind::Matmul ||
        candidate.kind == KernelKind::Elementwise ||
        candidate.kind == KernelKind::Generic) {
      op->removeAttr(secret::SecretDialect::kKernelAttrName);
    }
    return;
  }

  op->setAttr(secret::SecretDialect::kKernelAttrName,
              secret::KernelAttr::get(op->getContext(), *candidate.kernel,
                                      /*force=*/false));
}

void LayoutAssignment::markSelected(Value value, LayoutAttr layout) {
  if (!isTensorLike(value)) return;
  const Candidate* candidate = findCandidate(value, layout);
  if (!candidate) return;

  auto it = selectedLayouts.find(value);
  if (it != selectedLayouts.end()) {
    if (it->second == layout) {
      applySelectedKernel(value, *candidate);
      return;
    }

    const Candidate* existing = findCandidate(value, it->second);
    if (existing && isBetterCandidate(*existing, *candidate)) return;
  }
  selectedLayouts[value] = layout;
  applySelectedKernel(value, *candidate);

  LLVM_DEBUG(llvm::dbgs() << "Selected " << kernelKindName(candidate->kind)
                          << " candidate for value " << value << " with "
                          << layout << " at cost " << candidate->cost << "\n");

  for (auto [operand, operandLayout] :
       llvm::zip(candidate->operands, candidate->operandLayouts)) {
    markSelected(operand, operandLayout);
  }
}

SmallVector<Candidate> LayoutAssignment::candidatesForValue(Value value) {
  seedValue(value);
  auto it = candidates.find(value);
  if (it == candidates.end()) return {};
  return it->second;
}

SmallVector<Candidate> LayoutAssignment::chooseCommonOperandCandidates(
    Operation* op) {
  return chooseCommonOperandCandidates(op, KernelKind::PassThrough);
}

SmallVector<Candidate> LayoutAssignment::chooseCommonOperandCandidates(
    Operation* op, KernelKind kind) {
  SmallVector<Value> operands;
  SmallVector<SmallVector<Candidate>> candidateSets;
  for (Value operand : op->getOperands()) {
    if (!isTensorLike(operand)) continue;
    SmallVector<Candidate> operandCandidates = candidatesForValue(operand);
    if (operandCandidates.empty()) continue;
    operands.push_back(operand);
    candidateSets.push_back(operandCandidates);
  }
  return chooseCommonCandidates(
      operands, candidateSets, kind,
      [&](LayoutAttr layout) { return operationCost(op, layout); });
}

SmallVector<Candidate> LayoutAssignment::chooseAlignedElementwiseCandidates(
    ArrayRef<Value> operands, KernelKind kind,
    function_ref<int64_t(LayoutAttr)> localCostFn,
    std::optional<KernelName> rotomKernel) {
  if (operands.size() != 2) return {};

  auto lhsType =
      dyn_cast<RankedTensorType>(getPlainValueType(operands[0].getType()));
  auto rhsType =
      dyn_cast<RankedTensorType>(getPlainValueType(operands[1].getType()));
  if (!lhsType || !rhsType || lhsType.getRank() != rhsType.getRank()) {
    return {};
  }

  SmallVector<std::pair<int64_t, int64_t>> identityDims;
  identityDims.reserve(lhsType.getRank());
  for (int64_t dim = 0; dim < lhsType.getRank(); ++dim) {
    identityDims.push_back({dim, dim});
  }

  SmallVector<Candidate> lhsCandidates = candidatesForValue(operands[0]);
  SmallVector<Candidate> rhsCandidates = candidatesForValue(operands[1]);
  SmallVector<Value> operandValues(operands.begin(), operands.end());
  SmallVector<Candidate> chosen;
  for (const Candidate& lhsCandidate : lhsCandidates) {
    for (const Candidate& rhsCandidate : rhsCandidates) {
      bool aligned = layoutsAlignedByDimMap(lhsCandidate.layout,
                                            rhsCandidate.layout, identityDims);

      int64_t sharedCost = lhsCandidate.cost + rhsCandidate.cost +
                           localCostFn(lhsCandidate.layout);
      if (!aligned) {
        sharedCost +=
            layoutConversionCost(rhsCandidate.layout, lhsCandidate.layout);
      }
      std::optional<KernelName> lhsKernel;
      if (rotomKernel &&
          supportsRotomAlignmentLowering(
              lhsCandidate.layout, rhsCandidate.layout, lhsCandidate.layout)) {
        lhsKernel = rotomKernel;
      }
      chosen.push_back({lhsCandidate.layout,
                        sharedCost,
                        kind,
                        operandValues,
                        {lhsCandidate.layout, rhsCandidate.layout},
                        lhsKernel});

      if (rhsCandidate.layout == lhsCandidate.layout) continue;
      int64_t reverseCost = lhsCandidate.cost + rhsCandidate.cost +
                            localCostFn(rhsCandidate.layout);
      if (!aligned) {
        reverseCost +=
            layoutConversionCost(lhsCandidate.layout, rhsCandidate.layout);
      }
      std::optional<KernelName> rhsKernel;
      if (rotomKernel &&
          supportsRotomAlignmentLowering(
              lhsCandidate.layout, rhsCandidate.layout, rhsCandidate.layout)) {
        rhsKernel = rotomKernel;
      }
      chosen.push_back({rhsCandidate.layout,
                        reverseCost,
                        kind,
                        operandValues,
                        {lhsCandidate.layout, rhsCandidate.layout},
                        rhsKernel});
    }
  }
  return uniqueCandidates(chosen);
}

void LayoutAssignment::assignResultsFromCandidates(Operation* op,
                                                   ArrayRef<Candidate> chosen) {
  if (chosen.empty()) return;
  for (Value result : op->getResults()) {
    if (!isTensorLike(result)) continue;
    setCandidates(result, chosen);
  }
}

LogicalResult LayoutAssignment::visitFunc(func::FuncOp op) {
  for (Value arg : op.getArguments()) seedValue(arg);
  return success();
}

LogicalResult LayoutAssignment::visitReturn(func::ReturnOp op) {
  auto func = op->getParentOfType<func::FuncOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    std::optional<Candidate> candidate = bestCandidate(operand.get());
    if (!candidate) continue;
    func.setResultAttr(operand.getOperandNumber(), kRotomLayoutAttrName,
                       candidate->layout);
    markSelected(operand.get(), candidate->layout);
  }
  return success();
}

LogicalResult LayoutAssignment::visitGeneric(secret::GenericOp op) {
  for (OpOperand& operand : op->getOpOperands()) {
    SmallVector<Candidate> operandCandidates =
        candidatesForValue(operand.get());
    if (operandCandidates.empty()) continue;
    BlockArgument blockArg =
        op.getRegion().getArgument(operand.getOperandNumber());
    SmallVector<Candidate> blockArgCandidates;
    for (const Candidate& candidate : operandCandidates) {
      blockArgCandidates.push_back({candidate.layout,
                                    candidate.cost,
                                    KernelKind::BlockArgument,
                                    {operand.get()},
                                    {candidate.layout}});
    }
    setCandidates(blockArg, blockArgCandidates);
  }
  return success();
}

LogicalResult LayoutAssignment::visitYield(secret::YieldOp op) {
  auto generic = op->getParentOfType<secret::GenericOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    SmallVector<Candidate> yielded = candidatesForValue(operand.get());
    if (yielded.empty()) continue;
    SmallVector<Candidate> resultCandidates;
    for (const Candidate& candidate : yielded) {
      resultCandidates.push_back({candidate.layout,
                                  candidate.cost,
                                  KernelKind::Yield,
                                  {operand.get()},
                                  {candidate.layout}});
    }
    setCandidates(generic.getResult(operand.getOperandNumber()),
                  resultCandidates);
  }
  return success();
}

LogicalResult LayoutAssignment::visitPassThrough(Operation* op) {
  SmallVector<Candidate> chosen = chooseCommonOperandCandidates(op);
  if (chosen.empty()) {
    for (Value result : op->getResults()) seedValue(result);
    return success();
  }
  assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult LayoutAssignment::visitElementwise(Operation* op) {
  if (op->getNumOperands() == 2) {
    std::optional<KernelName> rotomKernel = selectRotomElementwiseKernel(op);
    SmallVector<Value> operands = {op->getOperand(0), op->getOperand(1)};
    SmallVector<Candidate> aligned = chooseAlignedElementwiseCandidates(
        operands, KernelKind::Elementwise,
        [&](LayoutAttr layout) { return operationCost(op, layout); },
        rotomKernel);
    if (!aligned.empty()) {
      assignResultsFromCandidates(op, aligned);
      return success();
    }
  }

  SmallVector<Candidate> chosen =
      chooseCommonOperandCandidates(op, KernelKind::Elementwise);
  assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult LayoutAssignment::visitGeneric(linalg::GenericOp op) {
  if (!isElementwiseGeneric(op)) return visitPassThrough(op);
  if (hasAddLikeBody(op) && op.getInputs().size() == 2) {
    SmallVector<Value> operands = {op.getInputs()[0], op.getInputs()[1]};
    SmallVector<Candidate> aligned = chooseAlignedElementwiseCandidates(
        operands, KernelKind::Generic,
        [&](LayoutAttr layout) { return genericOperationCost(op, layout); });
    if (!aligned.empty()) {
      assignResultsFromCandidates(op, aligned);
      return success();
    }
  }

  SmallVector<Value> operands;
  SmallVector<SmallVector<Candidate>> candidateSets;
  for (Value operand : op->getOperands()) {
    if (!isTensorLike(operand)) continue;
    SmallVector<Candidate> operandCandidates = candidatesForValue(operand);
    if (operandCandidates.empty()) continue;
    operands.push_back(operand);
    candidateSets.push_back(operandCandidates);
  }
  SmallVector<Candidate> chosen = chooseCommonCandidates(
      operands, candidateSets, KernelKind::Generic,
      [&](LayoutAttr layout) { return genericOperationCost(op, layout); });
  assignResultsFromCandidates(op, chosen);
  return success();
}

LogicalResult LayoutAssignment::visitMatmul(linalg::MatmulOp op) {
  if (op.getInputs().size() != 2 || op->getNumResults() != 1) {
    return visitPassThrough(op);
  }

  auto lhsType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || resultType.getRank() != 2) {
    return visitPassThrough(op);
  }

  Value lhs = op.getInputs()[0];
  Value rhs = op.getInputs()[1];
  Value init = op.getOutputs()[0];
  SmallVector<Candidate> lhsCandidates = candidatesForValue(lhs);
  SmallVector<Candidate> rhsCandidates = candidatesForValue(rhs);

  // Generate a ciphertext-axis diagonal (rolled) candidate for a square matvec,
  // so the search discovers the Halevi-Shoup baby-step/giant-step kernel without
  // a hand-specified rolled layout. This is the analog of the reference Rotom
  // apply_sum_roll (assignment/gen/align_rolls.py): it moves the contraction
  // onto the ciphertext axis (one ciphertext per diagonal) so the reduction
  // becomes a cheap ciphertext sum, then the rotation-aware cost above lets it
  // beat the row-major candidate. The diagonal layouts are added to the operand
  // candidate sets so the selected matmul can back-propagate them.
  std::optional<Candidate> diagonalCandidate;
  {
    int64_t kDim = lhsType.getDimSize(1);
    int64_t pDim = rhsType.getDimSize(1);
    int64_t mp = nextPowerOfTwo(lhsType.getDimSize(0));
    int64_t kp = nextPowerOfTwo(kDim);
    // Square (mp == kp) or squat (mp < kp, kp % mp == 0) matvec.
    if (pDim == 1 && kDim > 1 && rhsType.getDimSize(0) == kDim &&
        mp <= kp && kp % mp == 0 && !lhsCandidates.empty()) {
      int64_t n = lhsCandidates.front().layout.getN();
      MLIRContext* ctx = op.getContext();
      LayoutAttr diagLhs;
      LayoutAttr idRhs;
      if (n == kp || (n > kp && n % kp == 0)) {
        // Plain diagonal at the ciphertext size n: for n == K this is the
        // single-period packing (one diagonal per ciphertext); for n > K the
        // materializer's dim-straddle packs P = n/K diagonals per ciphertext
        // (dense). The vector and output occupy slots [0, K) and [0, M).
        diagLhs = makeCtDiagonalMatrixLayout(ctx, mp, kp, n);
        idRhs = makeIdentityVectorLayout(ctx, kp, n);
      }
      if (diagLhs && idRhs && isMaterializableRotomLayout(diagLhs) &&
          isMaterializableRotomLayout(idRhs) &&
          isLayoutCompatibleWithValue(diagLhs, lhs) &&
          isLayoutCompatibleWithValue(idRhs, rhs)) {
        // The diagonal result is masked back to the D output rows, so the output
        // is the non-replicated identity layout (not what combineMatmulOutputLayout
        // would derive from the replicated lhs).
        LayoutAttr outLayout = makeIdentityVectorLayout(ctx, mp, n);
        // Register the diagonal operand layouts so the selected matmul can
        // back-propagate them (markSelected -> findCandidate), without polluting
        // the row-major cross-product below. Only add a bare (operand-less)
        // candidate when the operand does NOT already carry that layout: an
        // intermediate operand (e.g. a prior op's result feeding this matvec's
        // vector) already has a candidate at this layout that records its own
        // operands, and a zero-cost bare duplicate would sort ahead of it and
        // sever back-propagation through that operand (findCandidate matches by
        // layout and returns the best-sorted match).
        auto hasLayout = [](ArrayRef<Candidate> cands, LayoutAttr layout) {
          return llvm::any_of(cands, [&](const Candidate& candidate) {
            return candidate.layout == layout;
          });
        };
        if (!hasLayout(lhsCandidates, diagLhs)) {
          SmallVector<Candidate> augLhs(lhsCandidates.begin(),
                                        lhsCandidates.end());
          augLhs.push_back({diagLhs, 0, KernelKind::Tensor});
          setCandidates(lhs, augLhs);
        }
        if (!hasLayout(rhsCandidates, idRhs)) {
          SmallVector<Candidate> augRhs(rhsCandidates.begin(),
                                        rhsCandidates.end());
          augRhs.push_back({idRhs, 0, KernelKind::Tensor});
          setCandidates(rhs, augRhs);
        }

        std::optional<KernelName> kernel = selectMatmulKernel(
            lhsType, rhsType, resultType, diagLhs, idRhs, outLayout);
        int64_t cost = matmulRotationCost(diagLhs, lhsType, rhsType);
        if (!kernel) cost += kUnloweredMatmulPenalty;
        diagonalCandidate = Candidate{outLayout,
                                      cost,
                                      KernelKind::Matmul,
                                      {lhs, rhs, init},
                                      {diagLhs, idRhs, outLayout},
                                      kernel};
      }
    }
  }

  SmallVector<Candidate> chosen;
  if (diagonalCandidate) chosen.push_back(*diagonalCandidate);
  for (const Candidate& lhsCandidate : lhsCandidates) {
    for (const Candidate& rhsCandidate : rhsCandidates) {
      std::optional<LayoutAttr> outputLayout =
          combineMatmulOutputLayout(lhsCandidate.layout, rhsCandidate.layout);
      if (!outputLayout) continue;

      int64_t cost = lhsCandidate.cost + rhsCandidate.cost +
                     matmulRotationCost(lhsCandidate.layout, lhsType, rhsType);
      if (!dimensionsAligned(lhsCandidate.layout, /*lhsDim=*/1,
                             rhsCandidate.layout, /*rhsDim=*/0)) {
        cost += layoutConversionCost(rhsCandidate.layout, lhsCandidate.layout);
      }
      std::optional<KernelName> kernel =
          selectMatmulKernel(lhsType, rhsType, resultType, lhsCandidate.layout,
                             rhsCandidate.layout, *outputLayout);
      // A matmul candidate with no kernel cannot be lowered (e.g. a row-major
      // multi-ciphertext matvec), so its true cost is infinite -- penalize it so
      // a kernel-bearing candidate always wins even when its rotation cost is
      // nominally higher (the small-matvec crossover).
      if (!kernel) cost += kUnloweredMatmulPenalty;
      chosen.push_back(
          {*outputLayout,
           cost,
           KernelKind::Matmul,
           {lhs, rhs, init},
           {lhsCandidate.layout, rhsCandidate.layout, *outputLayout},
           kernel});
    }
  }

  SmallVector<Candidate> matmulCandidates = uniqueCandidates(chosen);
  if (!matmulCandidates.empty()) {
    SmallVector<Candidate> initCandidates;
    for (const Candidate& candidate : matmulCandidates) {
      initCandidates.push_back({candidate.layout, 0, KernelKind::Matmul});
    }
    setCandidates(init, initCandidates);
  }
  assignResultsFromCandidates(op, matmulCandidates);
  return success();
}

LogicalResult LayoutAssignment::visitTranspose(linalg::TransposeOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputType) return visitPassThrough(op);

  SmallVector<int64_t> oldToNew(inputType.getRank(), -2);
  for (auto [outputDim, inputDim] : llvm::enumerate(op.getPermutation())) {
    if (inputDim < 0 || inputDim >= inputType.getRank()) {
      return visitPassThrough(op);
    }
    oldToNew[inputDim] = static_cast<int64_t>(outputDim);
  }

  SmallVector<Candidate> inputCandidates = candidatesForValue(op.getInput());
  SmallVector<Candidate> transposed = remapCandidates(
      op.getInput(), inputCandidates, oldToNew, KernelKind::Transpose);
  assignResultsFromCandidates(op, transposed);
  return success();
}

LogicalResult LayoutAssignment::visitReduction(linalg::ReduceOp op) {
  for (auto [input, result] : llvm::zip(op.getInputs(), op.getResults())) {
    SmallVector<Candidate> inputCandidates = candidatesForValue(input);
    if (inputCandidates.empty()) continue;

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) continue;

    std::optional<SmallVector<int64_t>> oldToNew =
        getReductionDimMap(inputType.getRank(), op.getDimensions());
    if (!oldToNew) continue;

    SmallVector<Candidate> reduced =
        remapCandidates(input, inputCandidates, *oldToNew, KernelKind::Reduce);
    for (Candidate& candidate : reduced) {
      candidate.cost += layoutNumCiphertexts(candidate.layout);
    }
    setCandidates(result, reduced);
  }
  return success();
}

LogicalResult LayoutAssignment::visitCollapseShape(tensor::CollapseShapeOp op) {
  std::optional<SmallVector<int64_t>> oldToNew =
      getCollapseShapeDimMap(op.getSrcType(), op.getReassociationIndices());
  if (!oldToNew) return visitPassThrough(op);

  SmallVector<Candidate> collapsed =
      remapCandidates(op.getSrc(), candidatesForValue(op.getSrc()), *oldToNew,
                      KernelKind::CollapseShape);
  assignResultsFromCandidates(op, collapsed);
  return success();
}

LogicalResult LayoutAssignment::visitExpandShape(tensor::ExpandShapeOp op) {
  std::optional<SmallVector<int64_t>> oldToNew =
      getExpandShapeDimMap(op.getResultType(), op.getReassociationIndices());
  if (!oldToNew) return visitPassThrough(op);

  SmallVector<Candidate> expanded =
      remapCandidates(op.getSrc(), candidatesForValue(op.getSrc()), *oldToNew,
                      KernelKind::ExpandShape);
  assignResultsFromCandidates(op, expanded);
  return success();
}

LogicalResult LayoutAssignment::visitExtractSlice(tensor::ExtractSliceOp op) {
  std::optional<SmallVector<int64_t>> oldToNew = getExtractSliceDimMap(
      op.getResultType(), op.getStaticSizes(), op.getStaticStrides());
  if (!oldToNew) return visitPassThrough(op);

  SmallVector<Candidate> sliced =
      remapCandidates(op.getSource(), candidatesForValue(op.getSource()),
                      *oldToNew, KernelKind::ExtractSlice);
  assignResultsFromCandidates(op, sliced);
  return success();
}

LogicalResult LayoutAssignment::visitInsertSlice(tensor::InsertSliceOp op) {
  SmallVector<Candidate> destCandidates = candidatesForValue(op.getDest());
  if (!destCandidates.empty()) {
    SmallVector<Candidate> sourceCandidates =
        candidatesForValue(op.getSource());
    std::optional<SmallVector<int64_t>> sourceToDest =
        getInsertSliceDimMap(op.getSourceType(), op.getResultType(),
                             op.getStaticSizes(), op.getStaticStrides());
    if (sourceToDest) {
      SmallVector<Candidate> expandedSource =
          remapCandidates(op.getSource(), sourceCandidates, *sourceToDest,
                          KernelKind::InsertSlice);
      if (!expandedSource.empty()) {
        SmallVector<Value> operands = {op.getDest(), op.getSource()};
        SmallVector<SmallVector<Candidate>> sets = {destCandidates,
                                                    expandedSource};
        assignResultsFromCandidates(
            op, chooseCommonCandidates(operands, sets, KernelKind::InsertSlice,
                                       [](LayoutAttr) { return 0; }));
        return success();
      }
    }
    assignResultsFromCandidates(op, destCandidates);
    return success();
  }

  std::optional<SmallVector<int64_t>> sourceToDest =
      getInsertSliceDimMap(op.getSourceType(), op.getResultType(),
                           op.getStaticSizes(), op.getStaticStrides());
  if (!sourceToDest) return visitPassThrough(op);

  SmallVector<Candidate> expandedSource =
      remapCandidates(op.getSource(), candidatesForValue(op.getSource()),
                      *sourceToDest, KernelKind::InsertSlice, /*extraCost=*/1);
  assignResultsFromCandidates(op, expandedSource);
  return success();
}

LogicalResult LayoutAssignment::visitOperation(Operation* op) {
  return TypeSwitch<Operation*, LogicalResult>(op)
      .Case<func::FuncOp>([&](auto typedOp) { return visitFunc(typedOp); })
      .Case<func::ReturnOp>([&](auto typedOp) { return visitReturn(typedOp); })
      .Case<secret::GenericOp>(
          [&](auto typedOp) { return visitGeneric(typedOp); })
      .Case<secret::YieldOp>([&](auto typedOp) { return visitYield(typedOp); })
      .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
            arith::MulFOp, arith::MulIOp>(
          [&](auto typedOp) { return visitElementwise(typedOp); })
      .Case<linalg::MatmulOp>(
          [&](auto typedOp) { return visitMatmul(typedOp); })
      .Case<linalg::GenericOp>(
          [&](auto typedOp) { return visitGeneric(typedOp); })
      .Case<linalg::TransposeOp>(
          [&](auto typedOp) { return visitTranspose(typedOp); })
      .Case<linalg::ReduceOp>(
          [&](auto typedOp) { return visitReduction(typedOp); })
      .Case<tensor::CollapseShapeOp>(
          [&](auto typedOp) { return visitCollapseShape(typedOp); })
      .Case<tensor::ExpandShapeOp>(
          [&](auto typedOp) { return visitExpandShape(typedOp); })
      .Case<tensor::ExtractSliceOp>(
          [&](auto typedOp) { return visitExtractSlice(typedOp); })
      .Case<tensor::InsertSliceOp>(
          [&](auto typedOp) { return visitInsertSlice(typedOp); })
      .Default(
          [&](Operation* genericOp) { return visitPassThrough(genericOp); });
}

void LayoutAssignment::writeSelectedLayouts() {
  for (auto& [value, layout] : selectedLayouts) {
    setAttributeAssociatedWith(value, kRotomLayoutAttrName, layout);
  }
}

void LayoutAssignment::runOnOperation() {
  ModuleOp module = getOperation();

  WalkResult result = module.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (failed(visitOperation(op))) return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  writeSelectedLayouts();
}

}  // namespace

}  // namespace mlir::heir::rotom
