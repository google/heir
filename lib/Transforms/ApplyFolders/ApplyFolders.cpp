#include "include/Transforms/ApplyFolders/ApplyFolders.h"

#include <utility>

#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_APPLYFOLDERS
#include "include/Transforms/ApplyFolders/ApplyFolders.h.inc"

struct ApplyFolders : impl::ApplyFoldersBase<ApplyFolders> {
  using ApplyFoldersBase::ApplyFoldersBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // No patterns added on purpose: this results in the greedy driver just
    // running folders.

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
