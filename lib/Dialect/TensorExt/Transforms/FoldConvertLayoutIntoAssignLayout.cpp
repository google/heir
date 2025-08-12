#include "lib/Dialect/TensorExt/Transforms/FoldConvertLayoutIntoAssignLayout.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project
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
                                PatternRewriter& rewriter) const override {
    if (op.getResult().getUsers().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    int64_t numConverted = 0;
    // Can't modify the users while iterating over them, so copy them to a
    // vector first.
    SmallVector<OpOperand, 4> users(op.getResult().getUsers());
    for (const OpOperand& user : users) {
      Operation* owner = user.getOwner();
      if (auto convertLayoutOp = dyn_cast<ConvertLayoutOp>(owner)) {
        if (convertLayoutOp.getFromLayout() != op.getLayout()) {
          // This should be considered invalid, but check again here for
          // safety.
          continue;
        }

        auto newOp = rewriter.replaceOpWithNewOp<AssignLayoutOp>(
            convertLayoutOp, op.getValue(), convertLayoutOp.getToLayout());
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
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldConvertLayout>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
