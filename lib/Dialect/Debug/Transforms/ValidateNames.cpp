#include "lib/Dialect/Debug/Transforms/ValidateNames.h"

#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "llvm/include/llvm/ADT/StringSet.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

namespace mlir {
namespace heir {
namespace debug {

#define GEN_PASS_DEF_DEBUGVALIDATENAMES
#include "lib/Dialect/Debug/Transforms/Passes.h.inc"

struct DebugValidateNames
    : public impl::DebugValidateNamesBase<DebugValidateNames> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringSet<> names;
    WalkResult result = module.walk([&](ValidateOp op) {
      StringRef name = op.getName();
      if (!names.insert(name).second) {
        op.emitError() << "duplicate debug.validate name: " << name;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace debug
}  // namespace heir
}  // namespace mlir
