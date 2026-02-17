#ifndef LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMEOPS_H_
#define LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMEOPS_H_

#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeDialect.h"
#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeOps.h.inc"

#endif  // LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMEOPS_H_
