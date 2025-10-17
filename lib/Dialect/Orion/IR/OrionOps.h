#ifndef LIB_DIALECT_ORION_IR_ORIONOPS_H_
#define LIB_DIALECT_ORION_IR_ORIONOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Orion/IR/OrionOps.h.inc"

#endif  // LIB_DIALECT_ORION_IR_ORIONOPS_H_
