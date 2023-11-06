#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_H_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_H_

#include "include/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "include/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/TfheRust/IR/TfheRustOps.h.inc"

#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_H_
