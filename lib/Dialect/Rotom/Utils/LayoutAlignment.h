#ifndef LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_

#include <cstdint>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

size_t inferCtPrefixLen(LayoutAttr layout);

int64_t layoutNumCiphertexts(LayoutAttr layout);

// A single bit of a tensor axis that occupies a different slot position in the
// two layouts and must be rotated to align them for an elementwise op.
struct ConversionMove {
  int64_t dim;       // tensor axis whose bit moves
  int64_t bit;       // which bit of that axis (log2 of the within-axis stride)
  int64_t fromSlot;  // its slot-bit position in `rhs`
  int64_t toSlot;    // its slot-bit position in `lhs`
};

// The slot bits that must move to align `rhs` onto `lhs` for an elementwise
// (identity-correspondence) op. Empty => already aligned: only the slot region
// is compared, so ciphertext-order differences are free and ignored. Layouts
// that pack an axis at different granularity -- e.g. [0:4:1] vs
// [0:2:2][0:2:1] -- decompose to the same bits and compare equal.
//
// These moves are the entries of a tensor_ext::Mapping (source -> target
// CtSlot); a later stage feeds them to the Vos-Vos-Erkin shift network for
// cost and emission.
//
// v1 analyzes layouts whose slot region is power-of-two *traversal* pieces over
// the same bit set. Anything outside that -- a slot gap/replication, a differing
// ct/slot partition (different numCt), or mismatched `n` -- returns a single
// sentinel move (`dim == -1`) meaning "conversion needed, not yet described",
// which still reads as "not aligned" via the empty/non-empty check.
SmallVector<ConversionMove> conversionMoves(LayoutAttr lhs, LayoutAttr rhs);

bool hasOnlyUnitStridedTraversalDims(LayoutAttr layout);

bool isMaterializableRotomLayout(LayoutAttr layout);

bool supportsRotomAlignmentLowering(LayoutAttr lhsLayout, LayoutAttr rhsLayout,
                                    LayoutAttr resultLayout);

// Returns a copy of `layout` whose `rolls` metadata is exactly `rolls`,
// replacing any rolls already present. Each pair (i, j) indexes into the
// layout's `dims` array and encodes `roll(i, j)`: it modifies the indices of
// `dims[i]` by the indices of `dims[j]` via modular addition. The result is
// constructed without verification; callers that need a materializable layout
// should check `isMaterializableRotomLayout`.
LayoutAttr withRolls(LayoutAttr layout,
                     ArrayRef<std::pair<int64_t, int64_t>> rolls);

// Enumerates the single-roll variants of `layout` that materialize to a valid
// Rotom layout. Each variant appends exactly one `roll(i, j)` to the layout's
// existing rolls, ranging over ordered pairs of distinct traversal dims i, j
// (`dim >= 0`, not gap/replication) with equal extent. Variants that fail to
// materialize -- for example a roll that references a ciphertext-side dim, which
// the slot-line lowering cannot express -- are omitted. The base (un-rolled)
// layout is not included in the result.
SmallVector<LayoutAttr> enumerateSingleRolls(LayoutAttr layout);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
