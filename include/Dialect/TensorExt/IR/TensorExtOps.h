#ifndef HEIR_INCLUDE_DIALECT_TensorExt_IR_TensorExtOPS_H_
#define HEIR_INCLUDE_DIALECT_TensorExt_IR_TensorExtOPS_H_

#include "include/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/TensorExt/IR/TensorExtOps.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_TensorExt_IR_TensorExtOPS_H_
