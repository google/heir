#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_H_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_H_

#include "include/Dialect/BGV/IR/BGVAttributes.h"
#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/BGV/IR/BGVTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/BGV/IR/BGVOps.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVOPS_H_
