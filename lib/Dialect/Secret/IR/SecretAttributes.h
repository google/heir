#ifndef LIB_DIALECT_SECRET_IR_SECRETATTRIBUTES_H_
#define LIB_DIALECT_SECRET_IR_SECRETATTRIBUTES_H_

// IWYU pragma: begin_keep
#include "lib/Kernel/Kernel.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Secret/IR/SecretAttributes.h.inc"

#endif  // LIB_DIALECT_SECRET_IR_SECRETATTRIBUTES_H_
