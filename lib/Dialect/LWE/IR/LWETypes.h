#ifndef LIB_DIALECT_LWE_IR_LWETYPES_H_
#define LIB_DIALECT_LWE_IR_LWETYPES_H_

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWETypes.h.inc"

namespace mlir {
namespace heir {

// just declaration here
void getAsmResultNames(Operation *op, ::mlir::OpAsmSetValueNameFn setNameFn);

namespace lwe {

std::string lweSuggestNameForType(Type type);

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_IR_LWETYPES_H_
