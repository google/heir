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

// Cost of bringing one operand to its expanded placement, via the shared
// conversion pricing (shift network when the ciphertext count is unchanged,
// explicit rotate/mask/accumulate steps when it changes). nullopt only when
// the expanded layout cannot be materialized at all.
static std::optional<int64_t> operandAlignmentCost(AssignmentContext& ctx,
                                                   LayoutAttr from,
                                                   LayoutAttr expanded) {
  if (from == expanded) return 0;
  if (!isMaterializableRotomLayout(expanded)) return std::nullopt;
  return ctx.cachedConversionCost(from, expanded);
}

// The full price of one plan for one operand-layout pairing: align both
// operands to their expanded placements, one multiply per compute
// ciphertext, then the k reduction. The single formula generateMatmul
// prices candidates with and selectMatmulPlan re-derives the winner with.
static std::optional<int64_t> matmulPlanCost(AssignmentContext& ctx,
                                             LayoutAttr lhs, LayoutAttr rhs,
                                             const MatmulPlan& plan) {
  std::optional<int64_t> lhsAlign =
      operandAlignmentCost(ctx, lhs, plan.expandedLhs);
  std::optional<int64_t> rhsAlign =
      operandAlignmentCost(ctx, rhs, plan.expandedRhs);
  if (!lhsAlign || !rhsAlign) return std::nullopt;
  const RotomCostModel& costModel = getCostModel();
  return *lhsAlign + *rhsAlign +
         layoutNumCiphertexts(plan.computeLayout) *
             costModel.ciphertextMultiply +
         plan.reduceRotations * costModel.rotation +
         plan.reduceAdds * costModel.add;
}

std::optional<MatmulPlan> selectMatmulPlan(AssignmentContext& ctx,
                                           LayoutAttr lhs, LayoutAttr rhs,
                                           LayoutAttr result) {
  std::optional<MatmulPlan> best;
  std::optional<int64_t> bestCost;
  for (const MatmulPlan& plan : enumerateMatmulPlans(lhs, rhs)) {
    if (plan.resultLayout != result) continue;
    std::optional<int64_t> cost = matmulPlanCost(ctx, lhs, rhs, plan);
    if (!cost) continue;
    if (!bestCost || *cost < *bestCost) {
      best = plan;
      bestCost = cost;
    }
  }
  return best;
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
        std::optional<int64_t> cost =
            matmulPlanCost(ctx, lhsCandidate.layout, rhsCandidate.layout, plan);
        if (!cost) continue;

        Candidate candidate;
        candidate.layout = plan.resultLayout;
        candidate.kind = KernelKind::Matmul;
        candidate.operands = {lhs, rhs};
        candidate.operandLayouts = {lhsCandidate.layout, rhsCandidate.layout};
        candidate.localCost = *cost;
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
