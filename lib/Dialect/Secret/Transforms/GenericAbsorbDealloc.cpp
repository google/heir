#include "lib/Dialect/Secret/Transforms/GenericAbsorbDealloc.h"

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETGENERICABSORBDEALLOC
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

struct GenericAbsorbDealloc
    : impl::SecretGenericAbsorbDeallocBase<GenericAbsorbDealloc> {
  using SecretGenericAbsorbDeallocBase::SecretGenericAbsorbDeallocBase;

  void runOnOperation() override {
    mlir::IRRewriter builder(&getContext());

    getOperation()->walk([&](secret::GenericOp op) {
      genericAbsorbDealloc(op, builder);
      return WalkResult::advance();
    });
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
