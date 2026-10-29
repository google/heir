#ifndef LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDKEYSWITCH_EXPANDKEYSWITCH_H_
#define LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDKEYSWITCH_EXPANDKEYSWITCH_H_

#include <memory>

#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DECL
#include "lib/Dialect/CheddarGPU/Transforms/ExpandKeySwitch/ExpandKeySwitch.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/CheddarGPU/Transforms/ExpandKeySwitch/ExpandKeySwitch.h.inc"

std::unique_ptr<mlir::Pass> createExpandKeySwitchPass();

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_TRANSFORMS_EXPANDKEYSWITCH_EXPANDKEYSWITCH_H_
