#ifndef INCLUDE_DIALECT_JAXITE_IR_JAXITEOPS_H_
#define INCLUDE_DIALECT_JAXITE_IR_JAXITEOPS_H_

#include "include/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "include/Dialect/Jaxite/IR/JaxiteTypes.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/Jaxite/IR/JaxiteOps.h.inc"

#endif  // INCLUDE_DIALECT_JAXITE_IR_JAXITEOPS_H_
