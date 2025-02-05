#ifndef LIB_DIALECT_LWE_IR_LWEATTRIBUTES_H_
#define LIB_DIALECT_LWE_IR_LWEATTRIBUTES_H_

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEEnums.h.inc"
#include "mlir/include/mlir/IR/TensorEncoding.h"  // from @llvm-project
// Required to pull in poly's Ring_Attr
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWEAttributes.h.inc"

#endif  // LIB_DIALECT_LWE_IR_LWEATTRIBUTES_H_
