#ifndef LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGOPS_H_
#define LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "mlir/include/mlir/Bytecode/BytecodeImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h.inc"

#endif  // LIB_DIALECT_PREPROCESSING_IR_PREPROCESSINGOPS_H_
