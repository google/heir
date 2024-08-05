#ifndef LIB_DIALECT_CKKS_IR_CKKSOPS_H_
#define LIB_DIALECT_CKKS_IR_CKKSOPS_H_

#include "lib/Dialect/LWE/IR/LWETraits.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/CKKS/IR/CKKSOps.h.inc"

#endif  // LIB_DIALECT_CKKS_IR_CKKSOPS_H_
