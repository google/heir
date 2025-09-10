#ifndef LIB_DIALECT_RNS_IR_RNSOPS_H_
#define LIB_DIALECT_RNS_IR_RNSOPS_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/RNS/IR/RNSOps.h.inc"

#endif  // LIB_DIALECT_RNS_IR_RNSOPS_H_
