#include "lib/Dialect/Openfhe/Transforms/HoistKeySwitchDown.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_HOISTKEYSWITCHDOWN
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

// Transforms: op(key_switch_down(a), key_switch_down(b))
//          => key_switch_down(op(a, b))
template <typename OpTy>
struct HoistKeySwitchThrough : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    auto lhsKs = op.getLhs().template getDefiningOp<KeySwitchDownOp>();
    auto rhsKs = op.getRhs().template getDefiningOp<KeySwitchDownOp>();

    if (!lhsKs || !rhsKs || !lhsKs->hasOneUse() || !rhsKs->hasOneUse())
      return failure();

    // create the operation in extended basis
    auto newOp =
        OpTy::create(rewriter, op.getLoc(), op.getType(), op.getCryptoContext(),
                     lhsKs.getCiphertext(), rhsKs.getCiphertext());

    auto newKs =
        KeySwitchDownOp::create(rewriter, op.getLoc(), op.getType(),
                                op.getCryptoContext(), newOp.getResult());

    rewriter.replaceOp(op, newKs.getResult());
    rewriter.eraseOp(lhsKs);
    rewriter.eraseOp(rhsKs);
    return success();
  }
};

struct HoistKeySwitchDown : impl::HoistKeySwitchDownBase<HoistKeySwitchDown> {
  using HoistKeySwitchDownBase::HoistKeySwitchDownBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    patterns.add<HoistKeySwitchThrough<AddOp>>(&getContext());
    patterns.add<HoistKeySwitchThrough<AddInPlaceOp>>(&getContext());
    patterns.add<HoistKeySwitchThrough<MulOp>>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
