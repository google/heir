#ifndef LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_H_
#define LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_H_

// NOLINTBEGIN(misc-include-cleaner): Required to define ArithExtOps
#include "lib/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

#define GET_OP_CLASSES
#include "lib/Dialect/ArithExt/IR/ArithExtOps.h.inc"

#endif  // LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_H_
