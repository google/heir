#ifndef LIB_DIALECT_HEIRINTERFACES_H_
#define LIB_DIALECT_HEIRINTERFACES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Don't mess up order
#include "lib/Dialect/HEIRInterfaces.h.inc"
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

void registerOperandAndResultAttrInterface(DialectRegistry& registry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_HEIRINTERFACES_H_
