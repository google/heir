#ifndef LIB_DIALECT_KEYMGMT_IR_KEYMGMTOPS_H_
#define LIB_DIALECT_KEYMGMT_IR_KEYMGMTOPS_H_

#include "lib/Dialect/KeyMgmt/IR/KeyMgmtDialect.h"
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/KeyMgmt/IR/KeyMgmtOps.h.inc"

#endif  // LIB_DIALECT_KEYMGMT_IR_KEYMGMTOPS_H_
