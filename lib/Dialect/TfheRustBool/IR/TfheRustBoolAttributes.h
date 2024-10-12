#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_H_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_H_

#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolEnums.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project

// Preserve import order
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolAttributes.h.inc"

#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_H_
