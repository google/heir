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
#include "llvm/include/llvm/Support/MathExtras.h"        // from @llvm-project
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

// Proxy cost of converting one operand's layout onto another, measured by the
// slot bits that must move. Zero when the layouts are already aligned (no
// moves). A later stage replaces this with the Vos-Vos-Erkin shift-network cost
// once the moves are lowered through tensor_ext.convert_layout.
int64_t conversionCost(ArrayRef<ConversionMove> moves) {
  if (moves.empty()) return 0;
  return 4 + static_cast<int64_t>(moves.size());
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

  void runOnOperation() override;

  // --- Candidate generation: forward pass that fills `candidates` for every
  // value. Each tensor op has its own visitor (visit*); these are kept
  // self-contained -- depending only on the shared seed/candidate/cost helpers
  // -- so that as the per-op kernel space grows they can move to one file each.
  LogicalResult generateCandidates(ModuleOp module);
  LogicalResult visitOperation(Operation* op);
  void seedValue(Value value);
  void setCandidates(Value value, ArrayRef<Candidate> newCandidates);
  SmallVector<Candidate> candidatesForValue(Value value);
  void assignResultsFromCandidates(Operation* op, ArrayRef<Candidate> chosen);
  SmallVector<Candidate> chooseCommonOperandCandidates(Operation* op);
  SmallVector<Candidate> chooseCommonOperandCandidates(Operation* op,
                                                       KernelKind kind);
  SmallVector<Candidate> chooseElementwiseKernels(
      ArrayRef<Value> operands, KernelKind kind,
      function_ref<int64_t(LayoutAttr)> computeCostFn,
      std::optional<KernelName> rotomKernel = std::nullopt);
  LogicalResult visitFunc(func::FuncOp op);
  LogicalResult visitGeneric(secret::GenericOp op);
  LogicalResult visitYield(secret::YieldOp op);
  LogicalResult visitPassThrough(Operation* op);
  LogicalResult visitElementwise(Operation* op);
  LogicalResult visitGeneric(linalg::GenericOp op);
  LogicalResult visitTranspose(linalg::TransposeOp op);
  LogicalResult visitReduction(linalg::ReduceOp op);
  LogicalResult visitCollapseShape(tensor::CollapseShapeOp op);
  LogicalResult visitExpandShape(tensor::ExpandShapeOp op);
  LogicalResult visitExtractSlice(tensor::ExtractSliceOp op);
  LogicalResult visitInsertSlice(tensor::InsertSliceOp op);

  // --- Layout search: choose one consistent assignment from the candidates
  // generated above, starting from each function's returned values. ---
  void selectLayouts(ModuleOp module);
  LogicalResult visitReturn(func::ReturnOp op);
  std::optional<Candidate> bestCandidate(Value value);
  const Candidate* findCandidate(Value value, LayoutAttr layout);
  void markSelected(Value value, LayoutAttr layout);
  void applySelectedKernel(Value value, const Candidate& candidate);
  void writeSelectedLayouts();
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
    if (candidate.kind == KernelKind::Elementwise ||
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

SmallVector<Candidate> LayoutAssignment::chooseElementwiseKernels(
    ArrayRef<Value> operands, KernelKind kind,
    function_ref<int64_t(LayoutAttr)> computeCostFn,
    std::optional<KernelName> rotomKernel) {
  if (operands.size() != 2) return {};

  auto lhsType =
      dyn_cast<RankedTensorType>(getPlainValueType(operands[0].getType()));
  auto rhsType =
      dyn_cast<RankedTensorType>(getPlainValueType(operands[1].getType()));
  if (!lhsType || !rhsType || lhsType.getRank() != rhsType.getRank()) {
    return {};
  }

  SmallVector<Candidate> lhsCandidates = candidatesForValue(operands[0]);
  SmallVector<Candidate> rhsCandidates = candidatesForValue(operands[1]);
  SmallVector<Value> operandValues(operands.begin(), operands.end());

  // Each (lhs, rhs) layout pairing yields up to two candidate kernels: compute
  // at the lhs layout (converting rhs onto it) or at the rhs layout (converting
  // lhs). A kernel first performs the conversion -- the slot bits
  // `conversionMoves` reports must move, free when none do -- then runs the
  // elementwise op at the shared compute layout.
  SmallVector<Candidate> kernels;
  for (const Candidate& lhs : lhsCandidates) {
    for (const Candidate& rhs : rhsCandidates) {
      auto addKernel = [&](LayoutAttr computeLayout, LayoutAttr convertFrom) {
        int64_t cost =
            lhs.cost + rhs.cost +
            conversionCost(conversionMoves(convertFrom, computeLayout)) +
            computeCostFn(computeLayout);
        std::optional<KernelName> kernel;
        if (rotomKernel && supportsRotomAlignmentLowering(
                               lhs.layout, rhs.layout, computeLayout)) {
          kernel = rotomKernel;
        }
        kernels.push_back({computeLayout, cost, kind, operandValues,
                           {lhs.layout, rhs.layout}, kernel});
      };
      addKernel(lhs.layout, rhs.layout);  // compute at lhs, convert rhs onto it
      if (rhs.layout != lhs.layout) {
        addKernel(rhs.layout, lhs.layout);  // compute at rhs, convert lhs onto it
      }
    }
  }
  return uniqueCandidates(kernels);
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
    SmallVector<Candidate> kernels = chooseElementwiseKernels(
        operands, KernelKind::Elementwise,
        [&](LayoutAttr layout) { return operationCost(op, layout); },
        rotomKernel);
    if (!kernels.empty()) {
      assignResultsFromCandidates(op, kernels);
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
    SmallVector<Candidate> kernels = chooseElementwiseKernels(
        operands, KernelKind::Generic,
        [&](LayoutAttr layout) { return genericOperationCost(op, layout); });
    if (!kernels.empty()) {
      assignResultsFromCandidates(op, kernels);
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
      .Case<secret::GenericOp>(
          [&](auto typedOp) { return visitGeneric(typedOp); })
      .Case<secret::YieldOp>([&](auto typedOp) { return visitYield(typedOp); })
      .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
            arith::MulFOp, arith::MulIOp>(
          [&](auto typedOp) { return visitElementwise(typedOp); })
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

LogicalResult LayoutAssignment::generateCandidates(ModuleOp module) {
  // Forward pre-order walk: every op contributes candidate layouts for its
  // results. Returns are search roots, not generators, so they are skipped.
  WalkResult result = module.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isa<func::ReturnOp>(op)) return WalkResult::advance();
    if (failed(visitOperation(op))) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void LayoutAssignment::selectLayouts(ModuleOp module) {
  // Backward search over the now-complete candidate space: from each returned
  // value pick the best candidate and propagate the choice to its producers.
  module.walk([&](func::ReturnOp op) { (void)visitReturn(op); });
}

void LayoutAssignment::runOnOperation() {
  ModuleOp module = getOperation();
  if (failed(generateCandidates(module))) {
    signalPassFailure();
    return;
  }
  selectLayouts(module);
  writeSelectedLayouts();
}

}  // namespace

}  // namespace mlir::heir::rotom
