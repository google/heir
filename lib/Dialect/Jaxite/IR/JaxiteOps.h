#ifndef LIB_DIALECT_JAXITE_IR_JAXITEOPS_H_
#define LIB_DIALECT_JAXITE_IR_JAXITEOPS_H_

#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteTypes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Jaxite/IR/JaxiteOps.h.inc"

#endif  // LIB_DIALECT_JAXITE_IR_JAXITEOPS_H_
