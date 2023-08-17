#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_H_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_H_

#include "include/Dialect/BGV/IR/BGVAttributes.h"
#include "include/Dialect/BGV/IR/BGVDialect.h"

// Required to pull in poly's Ring_Attr
#include "include/Dialect/Poly/IR/PolyAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "include/Dialect/BGV/IR/BGVTypes.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_H_
