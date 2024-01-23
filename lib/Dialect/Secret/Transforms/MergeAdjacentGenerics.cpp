#include "include/Dialect/Secret/Transforms/MergeAdjacentGenerics.h"

#include <utility>

#include "include/Dialect/Secret/IR/SecretPatterns.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETMERGEADJACENTGENERICS
#include "include/Dialect/Secret/Transforms/Passes.h.inc"

struct MergeAdjacentGenericsPass
    : impl::SecretMergeAdjacentGenericsBase<MergeAdjacentGenericsPass> {
  using SecretMergeAdjacentGenericsBase::SecretMergeAdjacentGenericsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    patterns.add<MergeAdjacentGenerics>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
