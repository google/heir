#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOMEMREF_PREPROCESSINGTOMEMREF_H_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOMEMREF_PREPROCESSINGTOMEMREF_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DECL
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToMemref/PreprocessingToMemref.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToMemref/PreprocessingToMemref.h.inc"

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOMEMREF_PREPROCESSINGTOMEMREF_H_
