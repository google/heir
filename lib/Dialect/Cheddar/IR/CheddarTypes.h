#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/ADT/StringRef.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"             // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h.inc"

namespace mlir {
namespace heir {
namespace cheddar {

// Attribute placed on function arguments carrying backend context, key
// material, or encoder state. Set either by the LWE-to-Cheddar lowering when
// threading support arguments into function signatures from the enclosing MLIR
// type, or after EmitC conversion from a C++ type name.
constexpr ::llvm::StringLiteral kSupportArgAttrName = "cheddar.support";

// The support kind of `type`, or empty when it is not a support type.
::llvm::StringRef getSupportKind(::mlir::Type type);

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_
