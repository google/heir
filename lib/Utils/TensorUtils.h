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

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_TENSORUTILS_H_
