#ifndef INCLUDE_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_H_
#define INCLUDE_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_H_

#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolOps.h.inc"

#endif  // INCLUDE_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_H_
