#ifndef LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

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
// All layouts here use iteration-space dim ids (below), NOT the operands'
// own tensor dims. Callers remap in and out (lhs (i,k): {0->kMatmulDimI,
// 1->kMatmulDimK}; rhs (k,j): {0->kMatmulDimK, 1->kMatmulDimJ}).
inline constexpr int64_t kMatmulDimI = 0;
inline constexpr int64_t kMatmulDimJ = 1;
inline constexpr int64_t kMatmulDimK = 2;

// One aligned placement of the iteration space. All four layouts share a
// piece-for-piece footprint with `computeLayout` (same sizes at the same
// positions), except `resultLayout`, which drops k's ciphertext pieces:
//   - expandedLhs: computeLayout with j-traversal pieces as replication --
//     the placement A must reach before the multiply;
//   - expandedRhs: computeLayout with i-traversal pieces as replication;
//   - computeLayout: the product's full (i, j, k) traversal placement;
//   - resultLayout: computeLayout after summing k -- a k piece in the slot
//     region becomes replication (rotate-and-reduce leaves the sum in every
//     k-slot); a k piece in the ciphertext prefix is dropped (ciphertext
//     adds collapse it).
//
// Costs are raw operation counts for the caller to weight with a cost model.
// Fill counts price replicating an operand's data into the replication slots
// it does not yet occupy: free across ciphertexts (a ciphertext copy is a
// handle copy), log2(extent) rotate-and-adds per distinct ciphertext within
// slots. They assume the operand arrives unreplicated; a candidate already
// matching the expanded layout owes conversion cost 0 and no fill (the
// caller's conversion pricing decides).
struct MatmulPlan {
  LayoutAttr expandedLhs;
  LayoutAttr expandedRhs;
  LayoutAttr computeLayout;
  LayoutAttr resultLayout;
  // Rotations/adds to fill the j-replication of expandedLhs.
  int64_t lhsFillRotations = 0;
  int64_t lhsFillAdds = 0;
  // Rotations/adds to fill the i-replication of expandedRhs.
  int64_t rhsFillRotations = 0;
  int64_t rhsFillAdds = 0;
  // Rotations/adds to sum k at computeLayout (rotate-and-reduce over k's
  // slot extent per surviving ciphertext, plus ciphertext adds for k's
  // ciphertext extent).
  int64_t reduceRotations = 0;
  int64_t reduceAdds = 0;
};

// Deterministically enumerates the aligned placements for one (lhs, rhs)
// layout pairing, both in iteration-space dims at the same n. Four
// candidates, deduplicated by computeLayout, each hosting one operand's
// placement unchanged and inserting the missing free dim as a whole
// unit-stride piece:
//   - lhs-hosted: lhs's (i, k) pieces, with j appended innermost (slot
//     placement) or prepended outermost (ciphertext placement);
//   - rhs-hosted: symmetric, inserting i.
// Variants whose layout fails LayoutAttr verification (e.g. a
// non-power-of-two extent landing in the slot region) are silently skipped,
// so fewer than four (possibly zero) plans may be returned. Materializability
// of the remapped per-value layouts is the caller's concern.
llvm::SmallVector<MatmulPlan> enumerateMatmulPlans(LayoutAttr lhs,
                                                   LayoutAttr rhs);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_
