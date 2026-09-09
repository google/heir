#ifndef LIB_DIALECT_ROTOM_IR_ROTOMOPS_H_
#define LIB_DIALECT_ROTOM_IR_ROTOMOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "mlir/include/mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Rotom/IR/RotomOps.h.inc"

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMOPS_H_
