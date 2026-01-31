#include "lib/Dialect/CKKS/Transforms/DecomposeKeySwitch.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/Transforms/Patterns.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DEF_DECOMPOSEKEYSWITCH
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

struct DecomposeKeySwitch : impl::DecomposeKeySwitchBase<DecomposeKeySwitch> {
  using DecomposeKeySwitchBase::DecomposeKeySwitchBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DecomposeKeySwitchPattern>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
