#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVATTRIBUTES_H_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVATTRIBUTES_H_

#include "include/Dialect/BGV/IR/BGVDialect.h"

// Required to pull in poly's Ring_Attr
#include "include/Dialect/Polynomial/IR/PolynomialAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "include/Dialect/BGV/IR/BGVAttributes.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVATTRIBUTES_H_
