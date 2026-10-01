#ifndef LIB_DIALECT_KERNEL_IR_KERNELTYPES_H_
#define LIB_DIALECT_KERNEL_IR_KERNELTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Kernel/IR/KernelTypes.h.inc"

#endif  // LIB_DIALECT_KERNEL_IR_KERNELTYPES_H_
