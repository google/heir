#ifndef LIB_DIALECT_POULPY_IR_POULPYOPS_H_
#define LIB_DIALECT_POULPY_IR_POULPYOPS_H_

#include "lib/Dialect/Poulpy/IR/PoulpyDialect.h"
#include "lib/Dialect/Poulpy/IR/PoulpyTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/Poulpy/IR/PoulpyOps.h.inc"

#endif  // LIB_DIALECT_POULPY_IR_POULPYOPS_H_
