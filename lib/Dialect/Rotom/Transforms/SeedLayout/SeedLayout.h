#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_SEEDLAYOUT_SEEDLAYOUT_H_
#define LIB_DIALECT_ROTOM_TRANSFORMS_SEEDLAYOUT_SEEDLAYOUT_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace rotom {

#define GEN_PASS_DECL_SEEDLAYOUT
#include "lib/Dialect/Rotom/Transforms/SeedLayout/SeedLayout.h.inc"

}  // namespace rotom
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_SEEDLAYOUT_SEEDLAYOUT_H_
