#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"

#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir::heir::rotom {

namespace {

bool hasOnlyUnitStrides(ArrayRef<int64_t> strides) {
  return llvm::all_of(strides, [](int64_t stride) { return stride == 1; });
}

}  // namespace

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
                                       KernelKind kind, int64_t extraCost) {
  SmallVector<Candidate> remapped;
  for (const Candidate& candidate : candidates) {
    std::optional<LayoutAttr> layout =
        remapLayoutDims(candidate.layout, oldToNewDim);
    if (!layout) continue;
    // The remap relabels dims of the operand's data (no rotation), so the
    // result inherits the operand's assignment and the only new local cost is
    // `extraCost`.
    Candidate result;
    result.layout = *layout;
    result.kind = kind;
    result.operands = {operand};
    result.operandLayouts = {candidate.layout};
    result.localCost = extraCost;
    result.assignment = candidate.assignment;
    result.accumulatedCost =
        accumulatedCostOf(result.assignment) + result.localCost;
    remapped.push_back(std::move(result));
  }
  return uniqueCandidates(remapped);
}

SmallVector<Candidate> chooseCommonCandidates(
    ArrayRef<Value> operands, ArrayRef<SmallVector<Candidate>> candidateSets,
    KernelKind kind, function_ref<int64_t(LayoutAttr)> localCostFn,
    function_ref<int64_t(LayoutAttr, LayoutAttr)> conversionCostFn) {
  if (operands.size() != candidateSets.size()) return {};

  SmallVector<Candidate> targets;
  for (const SmallVector<Candidate>& candidates : candidateSets) {
    for (const Candidate& candidate : candidates) {
      Candidate target;
      target.layout = candidate.layout;
      target.kind = kind;
      targets.push_back(std::move(target));
    }
  }
  targets = uniqueCandidates(targets);
  if (targets.empty()) return {};

  SmallVector<Candidate> chosen;
  for (const Candidate& target : targets) {
    int64_t localCost = localCostFn(target.layout);
    bool valid = true;
    Assignment assignment;
    SmallVector<LayoutAttr> operandLayouts;
    for (const SmallVector<Candidate>& candidates : candidateSets) {
      if (candidates.empty()) continue;
      const Candidate* bestCandidate = nullptr;
      std::optional<Candidate> bestScoredCandidate;
      int64_t bestConversion = 0;
      for (const Candidate& candidate : candidates) {
        int64_t conversion = conversionCostFn(candidate.layout, target.layout);
        Candidate scoredCandidate = candidate;
        scoredCandidate.accumulatedCost =
            candidate.accumulatedCost + conversion;
        scoredCandidate.kind = kind;
        if (!bestScoredCandidate ||
            isBetterCandidate(scoredCandidate, *bestScoredCandidate)) {
          bestScoredCandidate = scoredCandidate;
          bestCandidate = &candidate;
          bestConversion = conversion;
        }
      }
      if (!bestCandidate) {
        valid = false;
        break;
      }
      // Merge the chosen operand's assignment; a disagreement on a shared value
      // means this combination is inconsistent, so drop the target.
      if (!mergeAssignments(assignment, bestCandidate->assignment)) {
        valid = false;
        break;
      }
      localCost += bestConversion;
      operandLayouts.push_back(bestCandidate->layout);
    }
    if (!valid) continue;
    Candidate candidate;
    candidate.layout = target.layout;
    candidate.kind = kind;
    candidate.operands = SmallVector<Value>(operands);
    candidate.operandLayouts = operandLayouts;
    candidate.localCost = localCost;
    candidate.assignment = std::move(assignment);
    candidate.accumulatedCost =
        accumulatedCostOf(candidate.assignment) + candidate.localCost;
    chosen.push_back(std::move(candidate));
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

}  // namespace mlir::heir::rotom
