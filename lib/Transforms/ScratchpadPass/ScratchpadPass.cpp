#include "lib/Transforms/ScratchpadPass/ScratchpadPass.h"

#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SCRATCHPADPASS
#include "lib/Transforms/ScratchpadPass/ScratchpadPass.h.inc"

struct ScratchpadPass : impl::ScratchpadPassBase<ScratchpadPass> {
  using ScratchpadPassBase::ScratchpadPassBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    // patterns.add<>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
