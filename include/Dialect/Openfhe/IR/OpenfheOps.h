#ifndef INCLUDE_DIALECT_OPENFHE_IR_OPENFHEOPS_H_
#define INCLUDE_DIALECT_OPENFHE_IR_OPENFHEOPS_H_

#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "include/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/Openfhe/IR/OpenfheOps.h.inc"

#endif  // INCLUDE_DIALECT_OPENFHE_IR_OPENFHEOPS_H_
