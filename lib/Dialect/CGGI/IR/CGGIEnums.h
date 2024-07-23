#ifndef LIB_DIALECT_CGGI_IR_CGGIENUMS_H_
#define LIB_DIALECT_CGGI_IR_CGGIENUMS_H_

#include "llvm/include/llvm/ADT/StringRef.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/CGGI/IR/CGGIEnums.h.inc"

#endif  // LIB_DIALECT_CGGI_IR_CGGIENUMS_H_
