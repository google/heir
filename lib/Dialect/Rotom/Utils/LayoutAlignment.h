#ifndef LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
#define LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_

#include <cstdint>
#include <utility>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

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

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_UTILS_LAYOUTALIGNMENT_H_
