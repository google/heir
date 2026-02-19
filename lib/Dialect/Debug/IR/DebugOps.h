#ifndef LIB_DIALECT_DEBUG_IR_DEBUGOPS_H_
#define LIB_DIALECT_DEBUG_IR_DEBUGOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Debug/IR/DebugDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Debug/IR/DebugOps.h.inc"

#endif  // LIB_DIALECT_DEBUG_IR_DEBUGOPS_H_
