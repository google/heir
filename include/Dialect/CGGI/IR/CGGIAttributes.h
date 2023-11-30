#ifndef INCLUDE_DIALECT_CGGI_IR_CGGIATTRIBUTES_H_
#define INCLUDE_DIALECT_CGGI_IR_CGGIATTRIBUTES_H_

#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project

// Preserve import order
#define GET_ATTRDEF_CLASSES
#include "include/Dialect/CGGI/IR/CGGIAttributes.h.inc"

#endif  // INCLUDE_DIALECT_CGGI_IR_CGGIATTRIBUTES_H_
