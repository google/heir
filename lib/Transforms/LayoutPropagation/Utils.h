#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_

#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {

int64_t maxOfMaxes(::llvm::ArrayRef<int64_t> d1, ::llvm::ArrayRef<int64_t> d2);

// `dims` is a list of dims of a tensor, and `inserts` represents a list of
// inserted dims in that tensor (via alignment attr's insertedDims). This
// function shifts the values of `dims` as if the dims from `inserts` were
// inserted.
//
// Example:
//
// input = [0, 1, 2, 3]
// inserts = [1, 2]
//
// output = [0, 3, 4, 5]
//
// This allows you to track how `dims` maps to new dims as a result of the
// inserts.
::llvm::SmallVector<int64_t> shiftByInserted(::llvm::ArrayRef<int64_t> dims,
                                             ::llvm::ArrayRef<int64_t> inserts,
                                             bool increment = true);

// Map the dims in the `dims` list to new dims when the dims in `removed` are
// removed from the tensor.
//
// Assumes dims and removed do not have any common values
//
// Example:
// input = [0, 3, 4, 5]
// removed = [1, 2]
//
// output = [0, 1, 2, 3]
::llvm::SmallVector<int64_t> shiftByRemoved(::llvm::ArrayRef<int64_t> dims,
                                            ::llvm::ArrayRef<int64_t> removed);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_
