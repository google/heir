#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_H_
#define LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h.inc"

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_H_
