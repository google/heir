#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_

#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h.inc"

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
