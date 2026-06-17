#ifndef LIB_DIALECT_PREPROCESSING_TRANSFORMS_VALIDATEPREPROCESSING_H_
#define LIB_DIALECT_PREPROCESSING_TRANSFORMS_VALIDATEPREPROCESSING_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DECL_VALIDATEPREPROCESSING
#define GEN_PASS_DECL_PREPROCESSINGVALIDATE
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_VALIDATEPREPROCESSING_H_
