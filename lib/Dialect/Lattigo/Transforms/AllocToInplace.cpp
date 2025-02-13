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

    // InplaceOp has the form: output = InplaceOp(evaluator, lhs, rhs, inplace)
    // where inplace is the actual output but for SSA form we need to return a
    // new value
    rewriter.replaceOpWithNewOp<InplaceOp>(op, op.getOperand(1).getType(),
                                           op.getOperand(0), op.getOperand(1),
                                           op.getOperand(2), op.getOperand(1));
    return success();
  }

 private:
  Liveness *liveness;
};

template <typename UnaryOp, typename InplaceOp>
struct ConvertUnaryOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  ConvertUnaryOp(mlir::MLIRContext *context, Liveness *liveness)
      : OpRewritePattern<UnaryOp>(context), liveness(liveness) {}

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const override {
    // operand 0 is evaluator
    auto lhs = op.getOperand(1);
    if (!liveness->isDeadAfter(lhs, op)) {
      return failure();
    }

    // InplaceOp has the form: output = InplaceOp(evaluator, lhs, inplace)
    // where inplace is the actual output but for SSA form we need to return a
    // new value
    rewriter.replaceOpWithNewOp<InplaceOp>(op, op.getOperand(1).getType(),
                                           op.getOperand(0), op.getOperand(1),
                                           op.getOperand(1));
    return success();
  }

 private:
  Liveness *liveness;
};

template <typename RotateOp, typename InplaceOp>
struct ConvertRotateOp : public OpRewritePattern<RotateOp> {
  using OpRewritePattern<RotateOp>::OpRewritePattern;

  ConvertRotateOp(mlir::MLIRContext *context, Liveness *liveness)
      : OpRewritePattern<RotateOp>(context), liveness(liveness) {}

  LogicalResult matchAndRewrite(RotateOp op,
                                PatternRewriter &rewriter) const override {
    // operand 0 is evaluator
    auto lhs = op.getOperand(1);
    if (!liveness->isDeadAfter(lhs, op)) {
      return failure();
    }

    // InplaceOp has the form: output = InplaceOp(evaluator, lhs, inplace)
    // {offset} where inplace is the actual output but for SSA form we need to
    // return a new value
    rewriter.replaceOpWithNewOp<InplaceOp>(op, op.getOperand(1).getType(),
                                           op.getOperand(0), op.getOperand(1),
                                           op.getOperand(1), op.getOffset());
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

    patterns.add<
        ConvertBinOp<lattigo::BGVAddNewOp, lattigo::BGVAddOp>,
        ConvertBinOp<lattigo::BGVSubNewOp, lattigo::BGVSubOp>,
        ConvertBinOp<lattigo::BGVMulNewOp, lattigo::BGVMulOp>,
        ConvertUnaryOp<lattigo::BGVRelinearizeNewOp, lattigo::BGVRelinearizeOp>,
        ConvertUnaryOp<lattigo::BGVRescaleNewOp, lattigo::BGVRescaleOp>,
        ConvertRotateOp<lattigo::BGVRotateColumnsNewOp,
                        lattigo::BGVRotateColumnsOp> >(context, &liveness);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
