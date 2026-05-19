#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarOps.h.inc"

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
