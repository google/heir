#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_H_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_H_

#include "mlir/include/mlir/IR/BuiltinOps.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h" // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project

#include "include/Dialect/Poly/IR/PolyDialect.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"

#define GET_OP_CLASSES
#include "include/Dialect/Poly/IR/PolyOps.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_H_
