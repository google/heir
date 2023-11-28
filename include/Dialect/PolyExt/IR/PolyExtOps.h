#ifndef HEIR_INCLUDE_DIALECT_POLYEXT_IR_POLYEXTOPS_H_
#define HEIR_INCLUDE_DIALECT_POLYEXT_IR_POLYEXTOPS_H_

#include "include/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "include/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/PolyExt/IR/PolyExtOps.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_POLYEXT_IR_POLYEXTOPS_H_
