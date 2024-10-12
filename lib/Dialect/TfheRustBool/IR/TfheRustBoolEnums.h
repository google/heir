#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLENUMS_H_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLENUMS_H_

#include "llvm/include/llvm/ADT/StringRef.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project

// Preserve import order
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolEnums.h.inc"

#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLENUMS_H_
