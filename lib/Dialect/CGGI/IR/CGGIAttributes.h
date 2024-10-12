#ifndef LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_H_
#define LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_H_

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIEnums.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project

// Preserve import order
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CGGI/IR/CGGIAttributes.h.inc"

#endif  // LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_H_
