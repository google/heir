#include "lib/Dialect/Openfhe/Transforms/ConvertToExtendedBasis.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_CONVERTTOEXTENDEDBASIS
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

struct ConvertFastRotationOp : public OpRewritePattern<FastRotationOp> {
  using OpRewritePattern<FastRotationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FastRotationOp op,
                                PatternRewriter& rewriter) const override {
    auto fastRotExt = FastRotationExtOp::create(
        rewriter, op->getLoc(), op.getType(), op.getCryptoContext(),
        op.getInput(), op.getIndex(), op.getPrecomputedDigitDecomp(),
        /*addFirst=*/true);

    auto keySwitchDown =
        KeySwitchDownOp::create(rewriter, op->getLoc(), op.getType(),
                                op.getCryptoContext(), fastRotExt.getResult());

    rewriter.replaceOp(op, keySwitchDown.getResult());
    return success();
  }
};

struct ConvertToExtendedBasis
    : impl::ConvertToExtendedBasisBase<ConvertToExtendedBasis> {
  using ConvertToExtendedBasisBase::ConvertToExtendedBasisBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertFastRotationOp>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
