#ifndef LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Preprocessing/Transforms/ValidatePreprocessing.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_PASSES_H_
