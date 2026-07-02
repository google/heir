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
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir::rotom {

// Cost of bringing one operand to its expanded placement. Same ciphertext
// count: one convert_layout, priced by the shift network (which also fills
// slot-region replication). Different ciphertext count: the explicit
// rotate/mask/accumulate steps of planLayoutExpansion, priced exactly as the
// lowering emits them -- expensive expansions lose in the search on cost,
// not on capability. nullopt only when a layout cannot be materialized at
// all.
static std::optional<int64_t> operandAlignmentCost(AssignmentContext& ctx,
                                                   LayoutAttr from,
                                                   LayoutAttr expanded) {
  if (from == expanded) return 0;
  const RotomCostModel& costModel = getCostModel();
  if (layoutNumCiphertexts(from) == layoutNumCiphertexts(expanded)) {
    if (!isMaterializableRotomLayout(expanded)) return std::nullopt;
    return ctx.cachedConversionCost(from, expanded);
  }
  FailureOr<SmallVector<LayoutExpansionStep>> steps =
      planLayoutExpansion(from, expanded);
  if (failed(steps)) return std::nullopt;
  const int64_t n = from.getN();
  int64_t cost = 0;
  llvm::DenseSet<int64_t> targetsSeen;
  for (const LayoutExpansionStep& step : *steps) {
    if (step.shift != 0) cost += costModel.rotation;
    // A partial-row step needs a plaintext mask multiply (cheap; priced as
    // an add pending a dedicated plaintext-multiply weight).
    if (static_cast<int64_t>(step.targetSlots.size()) != n) {
      cost += costModel.add;
    }
    if (!targetsSeen.insert(step.targetCt).second) cost += costModel.add;
  }
  return cost;
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
