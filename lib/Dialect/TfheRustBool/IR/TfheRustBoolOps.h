#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_H_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_H_

#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolAttributes.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h.inc"

#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLOPS_H_
