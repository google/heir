#ifndef LIB_DIALECT_MGMT_IR_MGMTOPS_H_
#define LIB_DIALECT_MGMT_IR_MGMTOPS_H_

// NOLINTBEGIN(misc-include-cleaner): Required to define MgmtOps
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Mgmt/IR/MgmtOps.h.inc"

#endif  // LIB_DIALECT_MGMT_IR_MGMTOPS_H_
