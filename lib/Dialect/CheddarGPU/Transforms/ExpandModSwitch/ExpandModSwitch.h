#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDMODSWITCH_EXPANDMODSWITCH_H_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDMODSWITCH_EXPANDMODSWITCH_H_

#include <memory>

#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DECL
#include "lib/Dialect/CheddarGPU/Transforms/ExpandModSwitch/ExpandModSwitch.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CheddarGPU/Transforms/ExpandModSwitch/ExpandModSwitch.h.inc"

std::unique_ptr<mlir::Pass> createExpandModSwitchPass();

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDMODSWITCH_EXPANDMODSWITCH_H_
