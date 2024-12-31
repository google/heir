#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_H_
#define LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_H_

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h.inc"

namespace mlir {
namespace heir {

// just declaration here
void getAsmResultNames(Operation *op, ::mlir::OpAsmSetValueNameFn setNameFn);

namespace openfhe {

std::string openfheSuggestNameForType(Type type);

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHETYPES_H_
