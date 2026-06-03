#ifndef LIB_TRANSFORMS_LINALGFUSELINEAROPS_LINALGFUSELINEAROPS_H_
#define LIB_TRANSFORMS_LINALGFUSELINEAROPS_LINALGFUSELINEAROPS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

std::unique_ptr<Pass> createLinalgFuseLinearOpsPass();

#define GEN_PASS_DECL
#include "lib/Transforms/LinalgFuseLinearOps/LinalgFuseLinearOps.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LinalgFuseLinearOps/LinalgFuseLinearOps.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LINALGFUSELINEAROPS_LINALGFUSELINEAROPS_H_
