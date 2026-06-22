#ifndef LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

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
// This is the cheap "already aligned?" check (empty => no conversion). The
// actual rotation cost of a non-empty conversion comes from
// `shiftNetworkConversionCost`, which lowers both layouts to tensor_ext and runs
// the Vos-Vos-Erkin shift network.
//
// v1 analyzes layouts whose slot region is power-of-two *traversal* pieces over
// the same bit set. Anything outside that -- a slot gap/replication, a differing
// ct/slot partition (different numCt), or mismatched `n` -- returns a single
// sentinel move (`dim == -1`) meaning "conversion needed, not yet described",
// which still reads as "not aligned" via the empty/non-empty check.
SmallVector<ConversionMove> conversionMoves(LayoutAttr lhs, LayoutAttr rhs);

// The Vos-Vos-Erkin shift-network rotation cost of converting a value packed as
// `from` into `to` (rotom layouts of the same tensor at the same `n`). Lowers
// both layouts to tensor_ext and reuses `computeCostOfLayoutConversion`, which
// composes inverse(from) o to into a (ct, slot) mapping, finds a shift scheme,
// and counts rotations. Returns 0 when the layouts are equal and std::nullopt
// when either cannot be lowered to tensor_ext. Expensive -- callers should gate
// it behind the `conversionMoves(...).empty()` fast path and cache by layout
// pair.
std::optional<int64_t> shiftNetworkConversionCost(LayoutAttr from,
                                                  LayoutAttr to);

bool hasOnlyUnitStridedTraversalDims(LayoutAttr layout);

bool isMaterializableRotomLayout(LayoutAttr layout);

// Whether the current Rotom elementwise lowering can align two operand layouts
// onto a result layout (same `n`, unit-strided traversal dims, materializable,
// same ciphertext count).
//
// TODO: this is elementwise-specific. As other tensor operators gain compute
// kernels they will have different alignment goals (e.g. a matmul aligns on the
// contraction axis, not identity), so this should be decomposed into a per-op
// alignment predicate rather than one shared check.
bool supportsRotomAlignmentLowering(LayoutAttr lhsLayout, LayoutAttr rhsLayout,
                                    LayoutAttr resultLayout);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
