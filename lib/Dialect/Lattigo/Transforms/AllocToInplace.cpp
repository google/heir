#include "lib/Dialect/Lattigo/Transforms/AllocToInplace.h"

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "mlir/include/mlir/Analysis/Liveness.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

template <typename BinOp, typename InplaceOp>
struct ConvertBinOp : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  ConvertBinOp(mlir::MLIRContext *context, Liveness *liveness)
      : OpRewritePattern<BinOp>(context), liveness(liveness) {}

  LogicalResult matchAndRewrite(BinOp op,
                                PatternRewriter &rewriter) const override {
    // operand 0 is evaluator
    auto lhs = op.getOperand(1);
    if (!liveness->isDeadAfter(lhs, op)) {
      return failure();
    }

    rewriter.create<InplaceOp>(op.getLoc(), op.getOperand(0), op.getOperand(1),
                               op.getOperand(2));
    rewriter.replaceAllUsesWith(op, lhs);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  Liveness *liveness;
};

#define GEN_PASS_DEF_ALLOCTOINPLACE
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

struct AllocToInplace : impl::AllocToInplaceBase<AllocToInplace> {
  using AllocToInplaceBase::AllocToInplaceBase;

  void runOnOperation() override {
    Liveness liveness(getOperation());

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ConvertBinOp<lattigo::BGVAddNewOp, lattigo::BGVAddOp>,
                 ConvertBinOp<lattigo::BGVSubNewOp, lattigo::BGVSubOp>,
                 ConvertBinOp<lattigo::BGVMulNewOp, lattigo::BGVMulOp>>(
        context, &liveness);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
