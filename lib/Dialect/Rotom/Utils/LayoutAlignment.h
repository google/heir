#ifndef LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace presburger {
class IntegerRelation;
}  // namespace presburger

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
// `shiftNetworkConversionCost`, which lowers both layouts to tensor_ext and
// runs the Vos-Vos-Erkin shift network.
//
// v1 analyzes layouts whose slot region is power-of-two *traversal* pieces over
// the same bit set. Anything outside that -- a slot gap/replication, a
// differing ct/slot partition (different numCt), or mismatched `n` -- returns a
// single sentinel move (`dim == -1`) meaning "conversion needed, not yet
// described", which still reads as "not aligned" via the empty/non-empty check.
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

// One step of an explicit layout expansion: take ciphertext `sourceCt` of the
// source, rotate its slots left by `shift`, keep only `targetSlots`, and
// accumulate into ciphertext `targetCt` of the target. A step whose
// `targetSlots` cover all n slots needs no mask (a plain copy when the shift
// is also zero).
struct LayoutExpansionStep {
  int64_t targetCt;
  int64_t sourceCt;
  int64_t shift;
  llvm::SmallVector<int64_t> targetSlots;
};

// Decomposes the conversion of a value packed as `from` into `to` (rotom
// layouts of the same tensor at the same n, possibly with DIFFERENT
// ciphertext counts -- both expansion and compaction) into explicit
// rotate/mask/accumulate steps, grouped by (targetCt, sourceCt, shift) and
// deterministically ordered. This is the general path for conversions
// tensor_ext.convert_layout cannot express (its operand and result types must
// match); the layout assignment prices exactly these steps, so search prunes
// expensive expansions by cost rather than by capability. Fails when either
// layout does not lower to an ISL relation or the domains disagree.
FailureOr<SmallVector<LayoutExpansionStep>> planLayoutExpansion(LayoutAttr from,
                                                                LayoutAttr to);

// As above, on already-materialized tensor_ext layout relations (post
// materialization the rotom layouts are erased; the relations carry the same
// placement information). `n` is the ciphertext slot count.
FailureOr<SmallVector<LayoutExpansionStep>> planLayoutExpansion(
    const presburger::IntegerRelation& fromRelation,
    const presburger::IntegerRelation& toRelation, int64_t n);

// The chosen route for a ciphertext-count-PRESERVING conversion: either a
// tensor_ext.convert_layout lowered by the Vos-Vos-Erkin shift network, or
// the explicit rotate/mask/accumulate steps of planLayoutExpansion --
// whichever needs fewer rotations (ties keep the shift network). The VVE
// analysis sees only slot permutations, so it badly overprices structured
// fills like a rolled-by-replication placement (one whole-ciphertext
// rotation per block) that the step plan expresses directly. Layout
// assignment prices this choice and the materializer re-derives it from the
// same relations, so the priced route is the emitted route.
struct SameCountConversionChoice {
  bool useSteps;
  // Valid when useSteps: the plan, and its mask (plaintext-multiply + add)
  // and extra-accumulate counts.
  SmallVector<LayoutExpansionStep> steps;
  int64_t stepMasks = 0;
  int64_t stepAccumulates = 0;
  // Rotation count of the chosen route.
  int64_t rotations = 0;
};
FailureOr<SameCountConversionChoice> chooseSameCountConversion(
    tensor_ext::LayoutAttr from, tensor_ext::LayoutAttr to, int64_t n);

// Every verifier-legal single-roll variant of an unrolled layout: each
// unit-stride traversal piece rolled by each unit-stride
// traversal or replication piece elsewhere, extents equal or not (pairs
// entirely inside the
// ciphertext prefix are skipped -- rolling there only permutes ciphertext
// contents; gap partners are skipped -- a rolled-by gap claims its blocks
// and changes the ciphertext count). Layouts that already carry rolls yield
// nothing. Used to widen a source's candidate set with its diagonal
// packings: a rolled placement materializes every rotation of the rolled
// piece across its partner's blocks, so alignment against a consumer that
// wants those pieces swapped or shifted becomes block selection instead of
// slot permutation.
SmallVector<LayoutAttr> enumerateSingleRollVariants(LayoutAttr layout);

// Whether the current Rotom elementwise lowering can align two operand layouts
// onto a result layout (same `n`, unit-strided traversal dims,
// materializable). Differing ciphertext counts are allowed: same-count
// conversions lower to tensor_ext.convert_layout, count-changing ones to the
// explicit steps of planLayoutExpansion.
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
