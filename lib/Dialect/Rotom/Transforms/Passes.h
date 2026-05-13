#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"
#include "lib/Dialect/Rotom/Transforms/SeedLayout/SeedLayout.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace rotom {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Rotom/Transforms/SeedLayout/SeedLayout.h.inc"

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_PASSES_H_
