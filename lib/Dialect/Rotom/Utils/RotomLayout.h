#ifndef LIB_DIALECT_ROTOM_UTILS_ROTOMLAYOUT_H_
#define LIB_DIALECT_ROTOM_UTILS_ROTOMLAYOUT_H_

#include <cstdint>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

// A layout's dims as DimAttrs, for the helpers that take a piece list.
llvm::SmallVector<DimAttr> layoutDims(LayoutAttr layout);

// The number of distinct ciphertexts the layout occupies: the product of the
// ciphertext-side piece extents. A gap that is a roll's `by` argument claims
// its blocks (one rotation of the rolled piece per block index) and so counts;
// a plain gap is unclaimed space and does not.
int64_t layoutNumCiphertexts(LayoutAttr layout);

// The dimension-merged canonical form of `layout`: adjacent pieces that
// describe one contiguous piece ([0:2:8][0:8:1] -> [0:16:1]), adjacent
// replication and adjacent gaps merge, so equivalent forms of the same
// packing compare equal. The search dedupes candidates on this form. Merging
// keeps the address map: it never crosses the ct/slot boundary, and a piece a
// roll names never merges. Roll arguments are re-indexed through the merges.
LayoutAttr mergeAdjacentLayoutDims(LayoutAttr layout);

// Whether the layout lowers to an ISL relation, i.e. can be materialized as a
// tensor_ext layout.
bool isMaterializableRotomLayout(LayoutAttr layout);

// Whether the Rotom lowering can handle the layout at all: it materializes to
// a relation. A legality gate, not an alignment check -- differing ciphertext
// counts are fine, planLayoutConversion handles them.
bool isLowerableRotomLayout(LayoutAttr layout);

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_ROTOMLAYOUT_H_
