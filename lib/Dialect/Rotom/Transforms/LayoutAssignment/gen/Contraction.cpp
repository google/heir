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

// The compute-and-reduce price of one plan, exclusive of operand alignment:
// one multiply per compute ciphertext, then the k reduction.
static int64_t matmulKernelCost(const MatmulPlan& plan) {
  const RotomCostModel& costModel = getCostModel();
  return layoutNumCiphertexts(plan.computeLayout) *
             costModel.ciphertextMultiply +
         plan.reduceRotations * costModel.rotation +
         plan.reduceAdds * costModel.add;
}

// The full price of one plan for one operand-layout pairing: align both
// operands to their expanded placements, then the kernel. The single formula
// generateMatmul prices candidates with and selectMatmulPlan re-derives the
// winner with.
static std::optional<int64_t> matmulPlanCost(AssignmentContext& ctx,
                                             LayoutAttr lhs, LayoutAttr rhs,
                                             const MatmulPlan& plan) {
  std::optional<int64_t> lhsAlign =
      operandAlignmentCost(ctx, lhs, plan.expandedLhs);
  std::optional<int64_t> rhsAlign =
      operandAlignmentCost(ctx, rhs, plan.expandedRhs);
  if (!lhsAlign || !rhsAlign) return std::nullopt;
  return *lhsAlign + *rhsAlign + matmulKernelCost(plan);
}

// A source candidate is data packed at encode time rather than computed
// homomorphically: a seeded value (secret function argument or cleartext
// feeding secret compute), recognizable as a candidate straight from
// seedValue with no compute history. The encoder can pack such data directly
// at any materializable placement, so a plan's expanded operand placement is
// reachable for free by assigning it as the source's own layout instead of
// converting in ciphertext space.
static bool isSourceCandidate(const Candidate& candidate) {
  return candidate.kind == KernelKind::Tensor;
}

// The near-zero price of repacking a source at a demanded placement: enough
// to lose ties against a plan whose operands already sit at their seeded
// layouts (the user's packing stands when nothing beats it), and far below
// any real ciphertext conversion.
constexpr int64_t kSourceRepackCost = 1;

namespace {
// One way to bring an operand to its expanded placement: keep the operand at
// `layout` and pay `alignCost` in conversions, or (repack) assign the
// expanded placement to the source directly at encode time.
struct AlignOption {
  LayoutAttr layout;
  int64_t alignCost;
  bool repack;
};
}  // namespace

static SmallVector<AlignOption, 2> alignOptions(AssignmentContext& ctx,
                                                const Candidate& operand,
                                                LayoutAttr expanded) {
  SmallVector<AlignOption, 2> options;
  if (std::optional<int64_t> cost =
          operandAlignmentCost(ctx, operand.layout, expanded)) {
    options.push_back({operand.layout, *cost, /*repack=*/false});
  }
  // The repack option's price is carried on the source's own assignment
  // entry (see generateMatmul), so a source consumed by several ops at the
  // same repacked placement is charged once.
  if (isSourceCandidate(operand) && operand.layout != expanded &&
      isMaterializableRotomLayout(expanded)) {
    options.push_back({expanded, 0, /*repack=*/true});
  }
  return options;
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
        for (const AlignOption& lhsOption :
             alignOptions(ctx, lhsCandidate, plan.expandedLhs)) {
          for (const AlignOption& rhsOption :
               alignOptions(ctx, rhsCandidate, plan.expandedRhs)) {
            // A self-matmul shares one source: both sides must agree on its
            // single assigned layout.
            if (lhs == rhs && (lhsOption.repack != rhsOption.repack ||
                               lhsOption.layout != rhsOption.layout)) {
              continue;
            }

            Candidate candidate;
            candidate.layout = plan.resultLayout;
            candidate.kind = KernelKind::Matmul;
            candidate.operands = {lhs, rhs};
            candidate.operandLayouts = {lhsOption.layout, rhsOption.layout};
            candidate.localCost = lhsOption.alignCost + rhsOption.alignCost +
                                  matmulKernelCost(plan);
            candidate.assignment = merged;
            if (lhsOption.repack) {
              candidate.assignment[lhs] = {lhsOption.layout, kSourceRepackCost};
            }
            if (rhsOption.repack) {
              candidate.assignment[rhs] = {rhsOption.layout, kSourceRepackCost};
            }
            candidate.accumulatedCost =
                accumulatedCostOf(candidate.assignment) + candidate.localCost;
            candidates.push_back(std::move(candidate));
          }
        }
      }
    }
  }
  ctx.assignResultsFromCandidates(op, uniqueCandidates(candidates));
  return success();
}

}  // namespace mlir::heir::rotom
