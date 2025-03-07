#ifndef LIB_DIALECT_PISA_IR_PISAOPS_H_
#define LIB_DIALECT_PISA_IR_PISAOPS_H_

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"  // required for the type predicate we use
#include "lib/Dialect/PISA/IR/PISADialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/PISA/IR/PISAOps.h.inc"

#endif  // LIB_DIALECT_PISA_IR_PISAOPS_H_
