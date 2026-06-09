#ifndef LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGTYPES_H_
#define LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h.inc"

#endif  // LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGTYPES_H_
