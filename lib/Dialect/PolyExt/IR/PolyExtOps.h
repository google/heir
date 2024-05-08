#ifndef HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_H_
#define HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_H_

#include "lib/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/PolyExt/IR/PolyExtOps.h.inc"

#endif  // HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_H_
