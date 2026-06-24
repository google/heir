#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir::rotom {

LogicalResult generateCollapseShape(AssignmentContext& ctx,
                                    tensor::CollapseShapeOp op) {
  std::optional<SmallVector<int64_t>> oldToNew =
      getCollapseShapeDimMap(op.getSrcType(), op.getReassociationIndices());
  if (!oldToNew) return generatePassThrough(ctx, op);

  SmallVector<Candidate> collapsed =
      remapCandidates(op.getSrc(), ctx.candidatesForValue(op.getSrc()),
                      *oldToNew, KernelKind::CollapseShape);
  ctx.assignResultsFromCandidates(op, collapsed);
  return success();
}

LogicalResult generateExpandShape(AssignmentContext& ctx,
                                  tensor::ExpandShapeOp op) {
  std::optional<SmallVector<int64_t>> oldToNew =
      getExpandShapeDimMap(op.getResultType(), op.getReassociationIndices());
  if (!oldToNew) return generatePassThrough(ctx, op);

  SmallVector<Candidate> expanded =
      remapCandidates(op.getSrc(), ctx.candidatesForValue(op.getSrc()),
                      *oldToNew, KernelKind::ExpandShape);
  ctx.assignResultsFromCandidates(op, expanded);
  return success();
}

LogicalResult generateExtractSlice(AssignmentContext& ctx,
                                   tensor::ExtractSliceOp op) {
  std::optional<SmallVector<int64_t>> oldToNew = getExtractSliceDimMap(
      op.getResultType(), op.getStaticSizes(), op.getStaticStrides());
  if (!oldToNew) return generatePassThrough(ctx, op);

  SmallVector<Candidate> sliced =
      remapCandidates(op.getSource(), ctx.candidatesForValue(op.getSource()),
                      *oldToNew, KernelKind::ExtractSlice);
  ctx.assignResultsFromCandidates(op, sliced);
  return success();
}

LogicalResult generateInsertSlice(AssignmentContext& ctx,
                                  tensor::InsertSliceOp op) {
  SmallVector<Candidate> destCandidates = ctx.candidatesForValue(op.getDest());
  if (!destCandidates.empty()) {
    SmallVector<Candidate> sourceCandidates =
        ctx.candidatesForValue(op.getSource());
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
        ctx.assignResultsFromCandidates(
            op, chooseCommonCandidates(
                    operands, sets, KernelKind::InsertSlice,
                    [](LayoutAttr) { return 0; },
                    [&ctx](LayoutAttr from, LayoutAttr to) {
                      return ctx.cachedConversionCost(from, to);
                    }));
        return success();
      }
    }
    ctx.assignResultsFromCandidates(op, destCandidates);
    return success();
  }

  std::optional<SmallVector<int64_t>> sourceToDest =
      getInsertSliceDimMap(op.getSourceType(), op.getResultType(),
                           op.getStaticSizes(), op.getStaticStrides());
  if (!sourceToDest) return generatePassThrough(ctx, op);

  SmallVector<Candidate> expandedSource =
      remapCandidates(op.getSource(), ctx.candidatesForValue(op.getSource()),
                      *sourceToDest, KernelKind::InsertSlice, /*extraCost=*/1);
  ctx.assignResultsFromCandidates(op, expandedSource);
  return success();
}

}  // namespace mlir::heir::rotom
