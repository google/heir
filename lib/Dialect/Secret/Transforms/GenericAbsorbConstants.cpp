#include "lib/Dialect/Secret/Transforms/GenericAbsorbConstants.h"

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETGENERICABSORBCONSTANTS
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

struct GenericAbsorbConstants
    : impl::SecretGenericAbsorbConstantsBase<GenericAbsorbConstants> {
  using SecretGenericAbsorbConstantsBase::SecretGenericAbsorbConstantsBase;

  void runOnOperation() override {
    mlir::IRRewriter builder(&getContext());

    getOperation()->walk([&](secret::GenericOp op) {
      genericAbsorbConstants(op, builder);
      return WalkResult::advance();
    });
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
