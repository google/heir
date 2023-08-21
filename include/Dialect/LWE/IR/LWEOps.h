#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_H_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_H_

#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/LWE/IR/LWEOps.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEOPS_H_
