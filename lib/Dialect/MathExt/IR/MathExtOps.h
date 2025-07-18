#ifndef LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_H_
#define LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_H_

#include "lib/Dialect/MathExt/IR/MathExtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project-project

#define GET_OP_CLASSES
#include "lib/Dialect/MathExt/IR/MathExtOps.h.inc"

#endif  // LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_H_
