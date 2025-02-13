#ifndef LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_H_
#define LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_H_

#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h.inc"

#endif  // LIB_DIALECT_TFHERUST_IR_TFHERUSTOPS_H_
