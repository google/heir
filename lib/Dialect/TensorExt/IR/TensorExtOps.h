#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h.inc"

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
