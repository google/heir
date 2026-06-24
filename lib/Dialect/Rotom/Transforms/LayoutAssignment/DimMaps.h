#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_DIMMAPS_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_DIMMAPS_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir::heir::rotom {

// Rewrites a layout's traversal dims from the producer's index space into a
// consumer's index space using `oldToNewDim` (old dim -> new dim, or -1 to drop
// the dim). The dim map is total over the input's dims, so this never fails.
LayoutAttr remapLayoutDims(LayoutAttr layout, ArrayRef<int64_t> oldToNewDim);

// Per-op dim maps for shape-changing ops. Each returns `oldToNew`, mapping every
// input dim to its result dim (or -1 to drop it), or nullopt when the op isn't a
// pure dim relabel and so can't flow a layout through.

// linalg.reduce: drop the reduced dims, renumber the survivors.
std::optional<SmallVector<int64_t>> getReductionDimMap(
    int64_t inputRank, ArrayRef<int64_t> reductionDims);

// tensor.collapse_shape: each reassociation group keeps its one non-size-1 dim
// and drops the size-1 dims; nullopt if a group merges two real dims.
std::optional<SmallVector<int64_t>> getCollapseShapeDimMap(
    RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociationIndices);

// tensor.expand_shape (collapse's inverse): map each source dim to the one
// non-size-1 result dim of its reassociation group.
std::optional<SmallVector<int64_t>> getExpandShapeDimMap(
    RankedTensorType resultType,
    ArrayRef<ReassociationIndices> reassociationIndices);

// Rank-reducing tensor.extract_slice: full-extent source dims map to the
// matching result dims, dropped unit dims to -1. Unit strides only.
std::optional<SmallVector<int64_t>> getExtractSliceDimMap(
    RankedTensorType resultType, ArrayRef<int64_t> staticSizes,
    ArrayRef<int64_t> staticStrides);

// tensor.insert_slice source (extract's inverse): map each source dim to its
// matching result dim, skipping inserted unit dims. Unit strides only.
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
// `conversionCostFn(operandLayout, resultLayout)` to realign onto it plus
// `localCostFn(layout)` for the compute.
SmallVector<Candidate> chooseCommonCandidates(
    ArrayRef<Value> operands, ArrayRef<SmallVector<Candidate>> candidateSets,
    KernelKind kind, function_ref<int64_t(LayoutAttr)> localCostFn,
    function_ref<int64_t(LayoutAttr, LayoutAttr)> conversionCostFn);

}  // namespace mlir::heir::rotom

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_DIMMAPS_H_
