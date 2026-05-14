#ifndef LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_H_
#define LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.h"
#include "lib/Dialect/LWE/IR/LWETraits.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h.inc"

#endif  // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDOPS_H_
