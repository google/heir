#ifndef INCLUDE_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_H_
#define INCLUDE_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "include/Transforms/FullLoopUnroll/FullLoopUnroll.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Transforms/FullLoopUnroll/FullLoopUnroll.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_H_
