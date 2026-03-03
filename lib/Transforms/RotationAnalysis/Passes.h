#ifndef LIB_TRANSFORMS_ROTATIONANALYSIS_PASSES_H_
#define LIB_TRANSFORMS_ROTATIONANALYSIS_PASSES_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DECL_ROTATIONANALYSISPASS
#define GEN_PASS_REGISTRATION
#include "lib/Transforms/RotationAnalysis/Passes.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_ROTATIONANALYSIS_PASSES_H_
