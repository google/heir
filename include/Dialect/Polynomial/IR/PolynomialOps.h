#ifndef INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
#define INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_

#include "include/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "include/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/Polynomial/IR/PolynomialOps.h.inc"

#endif  // INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
