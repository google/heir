#ifndef LIB_DIALECT_BGV_IR_BGVOPS_H_
#define LIB_DIALECT_BGV_IR_BGVOPS_H_

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/LWE/IR/LWETraits.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/BGV/IR/BGVOps.h.inc"

#endif  // LIB_DIALECT_BGV_IR_BGVOPS_H_
