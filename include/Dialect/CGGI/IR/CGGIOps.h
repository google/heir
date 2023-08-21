#ifndef HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_H_
#define HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_H_

#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/CGGI/IR/CGGIOps.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_H_
