#ifndef HEIR_INCLUDE_DIALECT_LWE_IR_LWETYPES_H_
#define HEIR_INCLUDE_DIALECT_LWE_IR_LWETYPES_H_

#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/LWE/IR/LWEDialect.h"

// Required to pull in poly's Ring_Attr
#include "include/Dialect/Poly/IR/PolyAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "include/Dialect/LWE/IR/LWETypes.h.inc"

#endif  // HEIR_INCLUDE_DIALECT_LWE_IR_LWETYPES_H_
