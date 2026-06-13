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

bool dimensionsAligned(LayoutAttr lhsLayout, int64_t lhsDim,
                       LayoutAttr rhsLayout, int64_t rhsDim);

bool layoutsAlignedByDimMap(LayoutAttr lhsLayout, LayoutAttr rhsLayout,
                            ArrayRef<std::pair<int64_t, int64_t>> dimMap);

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
