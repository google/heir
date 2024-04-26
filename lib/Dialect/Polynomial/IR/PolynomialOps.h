#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_

#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h.inc"

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
