#ifndef INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
#define INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_

#include "include/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/TensorExt/IR/TensorExtOps.h.inc"

#endif  // INCLUDE_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
