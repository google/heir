#ifndef THIRD_PARTY_HEIR_INCLUDE_DIALECT_HEIR_IR_HEIROPS_H_
#define THIRD_PARTY_HEIR_INCLUDE_DIALECT_HEIR_IR_HEIROPS_H_

#include "mlir/include/mlir/IR/Attributes.h" // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h" // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h" // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h" // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h" // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project

// Get the C++ declaration for all the ops defined in ODS for the dialect.

#define GET_OP_CLASSES
#include "include/Dialect/HEIR/IR/HEIROps.h.inc"

#endif  // THIRD_PARTY_HEIR_INCLUDE_DIALECT_HEIR_IR_HEIROPS_H_
