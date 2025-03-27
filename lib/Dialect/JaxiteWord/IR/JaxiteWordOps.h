#ifndef LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_H_
#define LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_H_

#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h.inc"

#endif  // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_H_
