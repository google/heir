#include "lib/Transforms/ApplyFolders/ApplyFolders.h"

#include <utility>

#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_APPLYFOLDERS
#include "lib/Transforms/ApplyFolders/ApplyFolders.h.inc"

struct ApplyFolders : impl::ApplyFoldersBase<ApplyFolders> {
  using ApplyFoldersBase::ApplyFoldersBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    tensor::ControlConstantExtractSliceFusionFn controlFn =
        [](tensor::ExtractSliceOp op) { return true; };
    tensor::populateFoldConstantExtractSlicePatterns(patterns, controlFn);
    // Use the greedy pattern driver to apply folders.
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
