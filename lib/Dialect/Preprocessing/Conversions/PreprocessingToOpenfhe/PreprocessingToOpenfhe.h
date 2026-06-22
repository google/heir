#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOOPENFHE_PREPROCESSINGTOOPENFHE_H_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOOPENFHE_PREPROCESSINGTOOPENFHE_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DECL_PREPROCESSINGTOOPENFHE
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToOpenfhe/PreprocessingToOpenfhe.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToOpenfhe/PreprocessingToOpenfhe.h.inc"

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOOPENFHE_PREPROCESSINGTOOPENFHE_H_
