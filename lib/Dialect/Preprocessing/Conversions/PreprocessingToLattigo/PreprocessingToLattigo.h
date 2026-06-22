#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOLATTIGO_PREPROCESSINGTOLATTIGO_H_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOLATTIGO_PREPROCESSINGTOLATTIGO_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DECL_PREPROCESSINGTOLATTIGO
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToLattigo/PreprocessingToLattigo.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToLattigo/PreprocessingToLattigo.h.inc"

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOLATTIGO_PREPROCESSINGTOLATTIGO_H_
