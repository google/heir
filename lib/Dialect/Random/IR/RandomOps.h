#ifndef LIB_DIALECT_RANDOM_IR_RANDOMOPS_H_
#define LIB_DIALECT_RANDOM_IR_RANDOMOPS_H_

// NOLINTBEGIN(misc-include-cleaner): Required to define RandomOps
#include "lib/Dialect/Random/IR/RandomDialect.h"
#include "lib/Dialect/Random/IR/RandomTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// NOLINTEND(misc-include-cleaner)

#define GET_OP_CLASSES
#include "lib/Dialect/Random/IR/RandomOps.h.inc"

#endif  // LIB_DIALECT_RANDOM_IR_RANDOMOPS_H_
