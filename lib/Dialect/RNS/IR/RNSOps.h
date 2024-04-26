#ifndef LIB_DIALECT_RNS_IR_RNSOPS_H_
#define LIB_DIALECT_RNS_IR_RNSOPS_H_

#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/RNS/IR/RNSOps.h.inc"

#endif  // LIB_DIALECT_RNS_IR_RNSOPS_H_
