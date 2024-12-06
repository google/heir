#include "lib/Dialect/ModArith/Transforms/ConvertToMac.h"

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "mod-arith-mac"

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DEF_CONVERTTOMAC
#include "lib/Dialect/ModArith/Transforms/Passes.h.inc"

struct FindMac : public OpRewritePattern<mod_arith::AddOp> {
  FindMac(mlir::MLIRContext *context)
      : OpRewritePattern<mod_arith::AddOp>(context) {}

  LogicalResult matchAndRewrite(mod_arith::AddOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto *parent = op.getLhs().getDefiningOp();

    if (!parent || !isa<mod_arith::MulOp>(parent)) {
      return failure();
    }

    auto mul_parent = cast<mod_arith::MulOp>(parent);

    auto result =
        b.create<MacOp>(mul_parent.getLhs(), mul_parent.getRhs(), op.getRhs());

    rewriter.replaceOp(op, result);
    rewriter.eraseOp(mul_parent);

    return success();
  }
};

struct ConvertToMac : impl::ConvertToMacBase<ConvertToMac> {
  using ConvertToMacBase::ConvertToMacBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FindMac>(context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
