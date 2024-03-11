#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_H_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_H_

#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "mlir/include/mlir/IR/TensorEncoding.h"  // from @llvm-project

// Required to pull in poly's Ring_Attr
#include "include/Dialect/Polynomial/IR/PolynomialAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "include/Dialect/LWE/IR/LWEAttributes.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWEATTRIBUTES_H_
