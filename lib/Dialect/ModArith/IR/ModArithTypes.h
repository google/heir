#ifndef LIB_DIALECT_MODARITH_IR_MODARITHTYPES_H_
#define LIB_DIALECT_MODARITH_IR_MODARITHTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypeInterfaces.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/ModArith/IR/ModArithTypes.h.inc"

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHTYPES_H_
