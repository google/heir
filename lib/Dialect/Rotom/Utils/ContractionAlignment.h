#ifndef LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

// The roll-free matmul lowering plan: for C[i,j] = sum_k A[i,k] * B[k,j],
// align both operands into a shared placement of the (i, j, k) iteration
// space, multiply elementwise once, then sum the k pieces. A plan is a pure
// function of the two operand layouts -- there is no kernel; the layout
// assignment prices a plan and the lowering re-derives the identical plan
// from the assigned layouts.
//
// `computeLayout` uses the iteration-space dim ids below; every other layout
// in a plan uses its own tensor's dims (lhs (i,k) = (0,1); rhs (k,j) =
// (0,1); result (i,j) = (0,1)).
inline constexpr int64_t kMatmulDimI = 0;
inline constexpr int64_t kMatmulDimJ = 1;
inline constexpr int64_t kMatmulDimK = 2;

// The op attribute under which the layout assignment records a matmul's
// chosen (lhs, rhs, result) rotom layouts. The materializer erases the
// per-value rotom layout attributes, so the ciphertext lowering reads this
// to re-derive the plan.
inline constexpr llvm::StringLiteral kRotomMatmulAttrName = "rotom.matmul";

// One aligned placement of the iteration space. All four layouts share a
// piece-for-piece footprint with `computeLayout` (same sizes at the same
// positions), except `resultLayout`, which drops k's ciphertext pieces:
//   - expandedLhs: computeLayout with j-traversal pieces as replication --
//     the placement A must reach before the multiply (in A's own dims);
//   - expandedRhs: computeLayout with i-traversal pieces as replication (in
//     B's own dims);
//   - computeLayout: the product's full (i, j, k) traversal placement
//     (iteration-space dims);
//   - resultLayout: computeLayout after summing k -- a k piece in the slot
//     region becomes replication (rotate-and-reduce leaves the sum in every
//     k-slot); a k piece in the ciphertext prefix is dropped (ciphertext
//     adds collapse it) -- in C's own dims.
//
// The counts are informational raw operation counts (the generator prices
// operand alignment with the shift-network cost instead): fill counts price
// replicating an operand's data into its replication slots -- free across
// ciphertexts, log2(extent) rotate-and-adds per distinct ciphertext within
// slots; reduce counts price summing k at computeLayout.
struct MatmulPlan {
  LayoutAttr expandedLhs;
  LayoutAttr expandedRhs;
  LayoutAttr computeLayout;
  LayoutAttr resultLayout;
  int64_t lhsFillRotations = 0;
  int64_t lhsFillAdds = 0;
  int64_t rhsFillRotations = 0;
  int64_t rhsFillAdds = 0;
  int64_t reduceRotations = 0;
  int64_t reduceAdds = 0;
};

// Deterministically enumerates the aligned placements for one (lhs, rhs)
// layout pairing, both in their own tensor dims at the same n. Four
// candidates, deduplicated by computeLayout, each hosting one operand's
// placement unchanged and inserting the missing free dim as a whole
// unit-stride piece:
//   - lhs-hosted: lhs's (i, k) pieces, with j appended innermost (slot
//     placement) or prepended outermost (ciphertext placement);
//   - rhs-hosted: symmetric, inserting i.
// Variants whose layout fails LayoutAttr verification (e.g. a
// non-power-of-two extent landing in the slot region) are silently skipped,
// so fewer than four (possibly zero) plans may be returned.
llvm::SmallVector<MatmulPlan> enumerateMatmulPlans(LayoutAttr lhs,
                                                   LayoutAttr rhs);

// Strips the outermost run of ciphertext-region replication pieces (the
// part of an expanded placement realized by free ciphertext copies) and
// multiplies their extents into `replicationFactor`. Returns `layout`
// unchanged (factor 1) when the ciphertext prefix does not start with
// replication.
LayoutAttr stripOuterCtReplication(LayoutAttr layout,
                                   int64_t& replicationFactor);

// Whether the v1 ciphertext lowering can realize this plan for operands
// currently packed as `lhsSource`/`rhsSource`:
//   - per operand, the expanded placement must be reachable as one
//     same-shape layout conversion plus outermost ciphertext copies: any
//     ciphertext-region replication is a single outermost run, and the
//     stripped inner layout has the operand's own ciphertext count;
//   - k has at most one ciphertext piece and it is outermost, so the
//     ciphertext-side sum is a strided row-slice reduction.
// The layout assignment filters candidate plans with this predicate so it
// never assigns a layout combination the lowering cannot realize.
bool isLowerableMatmulPlan(const MatmulPlan& plan, LayoutAttr lhsSource,
                           LayoutAttr rhsSource);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_
