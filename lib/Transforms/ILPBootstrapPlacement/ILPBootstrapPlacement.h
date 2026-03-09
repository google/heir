#ifndef LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_H_
#define LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_H_
