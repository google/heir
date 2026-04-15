#ifndef LIB_DIALECT_RNS_IR_RNSTYPES_H_
#define LIB_DIALECT_RNS_IR_RNSTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/ModArith/IR/ModArithTypeInterfaces.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/RNS/IR/RNSTypes.h.inc"

#endif  // LIB_DIALECT_RNS_IR_RNSTYPES_H_
