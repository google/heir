#ifndef LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_H_
#define LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_H_

#include "lib/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferIntRangeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/ArithExt/IR/ArithExtOps.h.inc"

#endif  // LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_H_
