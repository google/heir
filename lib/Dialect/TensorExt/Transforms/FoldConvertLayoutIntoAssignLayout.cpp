#include "lib/Dialect/TensorExt/Transforms/FoldConvertLayoutIntoAssignLayout.h"

#include <utility>

#include "lib/Dialect/TensorExt/Transforms/Patterns.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_FOLDCONVERTLAYOUTINTOASSIGNLAYOUT
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

struct FoldConvertLayoutIntoAssignLayout
    : impl::FoldConvertLayoutIntoAssignLayoutBase<
          FoldConvertLayoutIntoAssignLayout> {
  using FoldConvertLayoutIntoAssignLayoutBase::
      FoldConvertLayoutIntoAssignLayoutBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldConvertLayoutIntoAssignLayoutPattern>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
