#ifndef HEIR_LIB_DIALECT_LWE_IR_LWEOPS_H_
#define HEIR_LIB_DIALECT_LWE_IR_LWEOPS_H_

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/LWE/IR/LWEOps.h.inc"

#endif  // HEIR_LIB_DIALECT_LWE_IR_LWEOPS_H_
