#ifndef LIB_DIALECT_ROTOM_IR_ROTOMDIALECT_H_
#define LIB_DIALECT_ROTOM_IR_ROTOMDIALECT_H_

#include <cstdint>

// IWYU pragma: begin_keep
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/Rotom/IR/RotomDialect.h.inc"

namespace mlir {
namespace heir {
namespace rotom {

// Marks an elementwise arith op in Rotom
inline constexpr llvm::StringLiteral kRotomElementwiseAttrName =
    "rotom.elementwise";

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMDIALECT_H_
