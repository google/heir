#ifndef LIB_UTILS_TENSORUTILS_H_
#define LIB_UTILS_TENSORUTILS_H_

#include <cstdint>

#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

// Returns the flattened index of the given tensor type and indices. Returns
// failure of the indices are not static values.
FailureOr<int64_t> getFlattenedIndex(RankedTensorType tensorType,
                                     SmallVector<OpFoldResult> indices);

// Given a row-major layout for the given shape, compute the list of indices
// that corresponds to the flattenedIndex.
SmallVector<int64_t> getIndicesFromRowMajorShape(int64_t flattenedIndex,
                                                 SmallVector<int64_t> shape);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_TENSORUTILS_H_
