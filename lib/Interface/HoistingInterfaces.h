#ifndef LIB_INTERFACE_HOISTINGINTERFACES_H_
#define LIB_INTERFACE_HOISTINGINTERFACES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
// IWYU pragma: end_keep

// IWYU pragma: begin_keep
#include "lib/Interface/HoistingOpInterfaces.h.inc"
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
// IWYU pragma: end_keep

#endif  // LIB_INTERFACE_HOISTINGINTERFACES_H_
