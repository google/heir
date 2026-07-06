#ifndef LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

// The Rotom matmul lowering plan: for C[i,j] = sum_k A[i,k] * B[k,j],
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
//     region becomes a gap (the cyclic rotate-and-reduce leaves the true sum
//     only at the k=0 offset; the other offsets hold unspecified window
//     sums); a k piece in the ciphertext prefix is dropped (ciphertext adds
//     collapse it); k rolls are consumed with their piece while i/j rolls
//     survive (a diagonal result) -- in C's own dims.
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
// layout pairing, both in their own tensor dims at the same n, deduplicated
// by computeLayout. Compute footprints host one operand's placement
// unchanged and place the missing free dim as a whole unit-stride piece:
//   - lhs-hosted: lhs's (i, k) pieces, with j appended innermost (slot
//     placement), prepended outermost (ciphertext placement), or replacing a
//     same-extent replication piece of the host (the reverse of expansion
//     subsumption, so an operand already at an expanded placement enumerates
//     the compute placement it came from);
//   - rhs-hosted: symmetric, placing i.
// Each footprint carries the host's own rolls (a rolled operand hosts its
// diagonal form at zero conversion) and yields the roll-free plan, the
// host-rolled plan, and single-roll decorations on either base (skipping
// pairs entirely inside the ciphertext prefix and any composition that
// would be order-dependent):
//   - a unit-stride k piece rolled by a same-extent unit-stride i/j piece:
//     a ciphertext-prefix k rolled by a slot piece is the ct-diagonal
//     family (k summed by plain ciphertext adds); a slot k rolled by a slot
//     piece is the Halevi-Shoup diagonal packing and by a ciphertext piece
//     the replicate-then-roll form (k summed by the usual slot
//     rotate-and-reduce);
//   - an i/j piece rolled by the other free/host traversal dim or by a
//     replication piece (the free-swap diagonal): the roll commutes with
//     the k-sum, so the result inherits it -- the plan produces a diagonal
//     result a downstream Halevi-Shoup matmul consumes at zero conversion.
// A non-k piece is never rolled by k (that would not commute with the
// k-sum). A rolled plan keeps the footprint (and so the multiply and
// reduction counts) unchanged; only the placements diagonalize.
// Variants whose layout fails LayoutAttr verification (e.g. a
// non-power-of-two extent landing in the slot region) are silently skipped,
// so possibly zero plans may be returned.
llvm::SmallVector<MatmulPlan> enumerateMatmulPlans(LayoutAttr lhs,
                                                   LayoutAttr rhs);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_CONTRACTIONALIGNMENT_H_
