#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_DIMMAPS_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_DIMMAPS_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                   // from @llvm-project

namespace mlir::heir::rotom {

// Rewrites a layout's traversal dims from the producer's index space into a
// consumer's index space using `oldToNewDim` (old dim -> new dim, or -1 to drop
// the dim). Returns nullopt if a dim cannot be mapped.
std::optional<LayoutAttr> remapLayoutDims(LayoutAttr layout,
                                          ArrayRef<int64_t> oldToNewDim);

// Per-op dim maps: each describes how a shape-changing op relabels the
// producer's tensor dims onto the result's dims (or -1 to drop). Returns nullopt
// when the op's shape is unsupported (dynamic, non-unit strides, ambiguous).
std::optional<SmallVector<int64_t>> getReductionDimMap(
    int64_t inputRank, ArrayRef<int64_t> reductionDims);
std::optional<SmallVector<int64_t>> getCollapseShapeDimMap(
    RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociationIndices);
std::optional<SmallVector<int64_t>> getExpandShapeDimMap(
    RankedTensorType resultType,
    ArrayRef<ReassociationIndices> reassociationIndices);
std::optional<SmallVector<int64_t>> getExtractSliceDimMap(
    RankedTensorType resultType, ArrayRef<int64_t> staticSizes,
    ArrayRef<int64_t> staticStrides);
std::optional<SmallVector<int64_t>> getInsertSliceDimMap(
    RankedTensorType sourceType, RankedTensorType resultType,
    ArrayRef<int64_t> staticSizes, ArrayRef<int64_t> staticStrides);

// Maps an operand's candidate layouts through `oldToNewDim` to produce the
// result's candidates (single-operand, shape-changing ops).
SmallVector<Candidate> remapCandidates(Value operand,
                                       ArrayRef<Candidate> candidates,
                                       ArrayRef<int64_t> oldToNewDim,
                                       KernelKind kind, int64_t extraCost = 0);

// Picks a common result layout across multiple operands, charging each operand
// the cost of converting onto it plus `localCostFn(layout)` for the compute.
SmallVector<Candidate> chooseCommonCandidates(
    ArrayRef<Value> operands, ArrayRef<SmallVector<Candidate>> candidateSets,
    KernelKind kind, function_ref<int64_t(LayoutAttr)> localCostFn);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_DIMMAPS_H_
