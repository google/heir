#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDHOIST_EXPANDHOIST_H_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDHOIST_EXPANDHOIST_H_

#include <memory>

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DECL
#include "lib/Dialect/CheddarGPU/Transforms/ExpandHoist/ExpandHoist.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CheddarGPU/Transforms/ExpandHoist/ExpandHoist.h.inc"

std::unique_ptr<mlir::Pass> createExpandHoistPass();

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDHOIST_EXPANDHOIST_H_
