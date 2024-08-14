#ifndef LIB_DIALECT_MODARITH_IR_MODARITHOPS_H_
#define LIB_DIALECT_MODARITH_IR_MODARITHOPS_H_

// NOLINTBEGIN(misc-include-cleaner): Required to define ModArithOps
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

#define GET_OP_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithOps.h.inc"

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHOPS_H_
