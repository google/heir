#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/AssignmentContext.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Candidate.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/Generators.h"
#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir::rotom {

// Cost of bringing one operand to its expanded placement: one same-shape
// layout conversion onto the inner placement (the shift network also fills
// slot-region replication), plus free outermost ciphertext copies. An
// operand already at the inner placement (e.g. a replicated seed) is free.
// nullopt when the inner placement cannot be materialized, which makes the
// plan unusable for this operand.
static std::optional<int64_t> operandAlignmentCost(AssignmentContext& ctx,
                                                   LayoutAttr from,
                                                   LayoutAttr expanded) {
  int64_t ctCopies = 1;
  LayoutAttr inner = stripOuterCtReplication(expanded, ctCopies);
  if (from == inner) return 0;
  if (!isMaterializableRotomLayout(inner)) return std::nullopt;
  return ctx.cachedConversionCost(from, inner);
}

LogicalResult generateMatmul(AssignmentContext& ctx, linalg::MatmulOp op) {
  // Only the default (i,k) x (k,j) indexing; transposed/broadcast variants
  // pass through until they get their own dim maps.
  if (op.hasUserDefinedMaps()) return generatePassThrough(ctx, op);

  Value lhs = op.getInputs()[0];
  Value rhs = op.getInputs()[1];
  SmallVector<Candidate> lhsCandidates = ctx.candidatesForValue(lhs);
  SmallVector<Candidate> rhsCandidates = ctx.candidatesForValue(rhs);

  // The init accumulator contributes no layout choice (like
  // generateReduction's inits): the lowering requires a zero fill.
  const RotomCostModel& costModel = getCostModel();
  SmallVector<Candidate> candidates;
  for (const Candidate& lhsCandidate : lhsCandidates) {
    for (const Candidate& rhsCandidate : rhsCandidates) {
      Assignment merged;
      if (!mergeAssignments(merged, lhsCandidate.assignment) ||
          !mergeAssignments(merged, rhsCandidate.assignment)) {
        continue;
      }
      for (const MatmulPlan& plan :
           enumerateMatmulPlans(lhsCandidate.layout, rhsCandidate.layout)) {
        // Only price plans the lowering can realize, so the selected
        // assignment is always executable.
        if (!isLowerableMatmulPlan(plan, lhsCandidate.layout,
                                   rhsCandidate.layout)) {
          continue;
        }
        std::optional<int64_t> lhsAlign =
            operandAlignmentCost(ctx, lhsCandidate.layout, plan.expandedLhs);
        std::optional<int64_t> rhsAlign =
            operandAlignmentCost(ctx, rhsCandidate.layout, plan.expandedRhs);
        if (!lhsAlign || !rhsAlign) continue;

        Candidate candidate;
        candidate.layout = plan.resultLayout;
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
