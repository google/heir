#ifndef LIB_DIALECT_PREPROCESSING_TRANSFORMS_USEDYNAMICRESOURCEDIR_H_
#define LIB_DIALECT_PREPROCESSING_TRANSFORMS_USEDYNAMICRESOURCEDIR_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DECL_USEDYNAMICRESOURCEDIR
#define GEN_PASS_DECL_PREPROCESSINGUSEDYNAMICRESOURCEDIR
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_USEDYNAMICRESOURCEDIR_H_
