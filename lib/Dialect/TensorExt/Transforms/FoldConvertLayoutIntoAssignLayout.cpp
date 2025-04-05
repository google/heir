#include "lib/Dialect/TensorExt/Transforms/FoldConvertLayoutIntoAssignLayout.h"

#include <utility>

#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_FOLDCONVERTLAYOUTINTOASSIGNLAYOUT
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

struct FoldConvertLayout : public OpRewritePattern<AssignLayoutOp> {
  using OpRewritePattern<AssignLayoutOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(AssignLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResult().getUsers().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    int64_t numConverted = 0;
    for (auto *user : op.getResult().getUsers()) {
      if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(user)) {
        if (convertLayoutOp.getFromLayout() != op.getLayout()) {
          // This should be considered invalid, but check again here for
          // safety.
          continue;
        }

        auto newOp = rewriter.replaceOpWithNewOp<AssignLayoutOp>(
            user, op.getValue(), convertLayoutOp.getToLayout());
        // Ensure the newOp has its layout attribute properly set
        setAttributeAssociatedWith(newOp.getResult(),
                                   TensorExtDialect::kLayoutAttrName,
                                   newOp.getLayout());
        ++numConverted;
      }
    }

    return numConverted > 0 ? success() : failure();
  }
};

struct FoldConvertLayoutIntoAssignLayout
    : impl::FoldConvertLayoutIntoAssignLayoutBase<
          FoldConvertLayoutIntoAssignLayout> {
  using FoldConvertLayoutIntoAssignLayoutBase::
      FoldConvertLayoutIntoAssignLayoutBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldConvertLayout>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
