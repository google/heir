#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir::rotom {

// The operand's pre-fill placement: replication pieces read as gaps, so the
// data footprint matches the unexpanded operand and the shift network can
// price the move onto it.
static LayoutAttr replicationAsGaps(LayoutAttr layout) {
  MLIRContext* ctx = layout.getContext();
  SmallVector<Attribute> dims;
  for (Attribute attr : layout.getDims()) {
    auto piece = cast<DimAttr>(attr);
    dims.push_back(piece.isReplicate()
                       ? DimAttr::get(ctx, /*dim=*/-2, piece.getSize(),
                                      /*stride=*/1)
                       : Attribute(piece));
  }
  return LayoutAttr::get(ctx, ArrayAttr::get(ctx, dims), layout.getN());
}

// Cost of bringing one operand to its expanded placement: convert onto the
// pre-fill placement, then rotate-and-add the replication slots full. An
// operand already at the expanded placement (e.g. a replicated seed) is
// free. nullopt when the pre-fill placement cannot be materialized, which
// makes the plan unusable for this operand.
static std::optional<int64_t> operandAlignmentCost(AssignmentContext& ctx,
                                                   LayoutAttr from,
                                                   LayoutAttr expanded,
                                                   int64_t fillRotations,
                                                   int64_t fillAdds) {
  if (from == expanded) return 0;
  LayoutAttr preFill = replicationAsGaps(expanded);
  if (!isMaterializableRotomLayout(preFill)) return std::nullopt;
  const RotomCostModel& costModel = getCostModel();
  return ctx.cachedConversionCost(from, preFill) +
         fillRotations * costModel.rotation + fillAdds * costModel.add;
}

LogicalResult generateMatmul(AssignmentContext& ctx, linalg::MatmulOp op) {
  // Only the default (i,k) x (k,j) indexing; transposed/broadcast variants
  // pass through until they get their own dim maps.
  if (op.hasUserDefinedMaps()) return generatePassThrough(ctx, op);

  Value lhs = op.getInputs()[0];
  Value rhs = op.getInputs()[1];
  SmallVector<Candidate> lhsCandidates = ctx.candidatesForValue(lhs);
  SmallVector<Candidate> rhsCandidates = ctx.candidatesForValue(rhs);

  // Tensor-dim <-> iteration-dim relabelings for C[i,j] = sum_k A[i,k]*B[k,j].
  // The init accumulator contributes no layout choice (like
  // generateReduction's inits): the lowering adds it at the result layout.
  const SmallVector<int64_t> lhsToIter = {kMatmulDimI, kMatmulDimK};
  const SmallVector<int64_t> rhsToIter = {kMatmulDimK, kMatmulDimJ};
  const SmallVector<int64_t> iterToLhs = {0, -1, 1};
  const SmallVector<int64_t> iterToRhs = {-1, 1, 0};
  const SmallVector<int64_t> iterToResult = {0, 1, -1};

  const RotomCostModel& costModel = getCostModel();
  SmallVector<Candidate> candidates;
  for (const Candidate& lhsCandidate : lhsCandidates) {
    for (const Candidate& rhsCandidate : rhsCandidates) {
      Assignment merged;
      if (!mergeAssignments(merged, lhsCandidate.assignment) ||
          !mergeAssignments(merged, rhsCandidate.assignment)) {
        continue;
      }
      LayoutAttr lhsIter = remapLayoutDims(lhsCandidate.layout, lhsToIter);
      LayoutAttr rhsIter = remapLayoutDims(rhsCandidate.layout, rhsToIter);
      for (const MatmulPlan& plan : enumerateMatmulPlans(lhsIter, rhsIter)) {
        std::optional<int64_t> lhsAlign =
            operandAlignmentCost(ctx, lhsCandidate.layout,
                                 remapLayoutDims(plan.expandedLhs, iterToLhs),
                                 plan.lhsFillRotations, plan.lhsFillAdds);
        std::optional<int64_t> rhsAlign =
            operandAlignmentCost(ctx, rhsCandidate.layout,
                                 remapLayoutDims(plan.expandedRhs, iterToRhs),
                                 plan.rhsFillRotations, plan.rhsFillAdds);
        if (!lhsAlign || !rhsAlign) continue;

        Candidate candidate;
        candidate.layout = remapLayoutDims(plan.resultLayout, iterToResult);
        candidate.kind = KernelKind::Matmul;
        candidate.operands = {lhs, rhs};
        candidate.operandLayouts = {lhsCandidate.layout, rhsCandidate.layout};
        candidate.localCost = *lhsAlign + *rhsAlign +
                              layoutNumCiphertexts(plan.computeLayout) *
                                  costModel.ciphertextMultiply +
                              plan.reduceRotations * costModel.rotation +
                              plan.reduceAdds * costModel.add;
        candidate.assignment = merged;
        candidate.accumulatedCost =
            accumulatedCostOf(candidate.assignment) + candidate.localCost;
        candidates.push_back(std::move(candidate));
      }
    }
  }
  ctx.assignResultsFromCandidates(op, uniqueCandidates(candidates));
  return success();
}

}  // namespace mlir::heir::rotom
