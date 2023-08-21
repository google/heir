#ifndef HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETOPS_H_
#define HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETOPS_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

// don't clobber import order
#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"

#define GET_OP_CLASSES
#include "include/Dialect/Secret/IR/SecretOps.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETOPS_H_
