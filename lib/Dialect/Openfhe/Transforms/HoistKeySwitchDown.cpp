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

// Pushes key_switch_down past binary operations to reduce key-switch count.
// Matches when at least one operand is key_switch_down (with single use).
// Example: add(key_switch_down(a), b) => key_switch_down(add(a, b))
template <typename OpTy>
struct HoistKeySwitchThrough : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    auto lhsKs = op.getLhs().template getDefiningOp<KeySwitchDownOp>();
    auto rhsKs = op.getRhs().template getDefiningOp<KeySwitchDownOp>();

    if (!lhsKs && !rhsKs) return failure();
    if (lhsKs && !lhsKs->hasOneUse()) return failure();
    if (rhsKs && !rhsKs->hasOneUse()) return failure();

    Value newLhs = lhsKs ? lhsKs.getCiphertext() : op.getLhs();
    Value newRhs = rhsKs ? rhsKs.getCiphertext() : op.getRhs();

    auto newOp = OpTy::create(rewriter, op.getLoc(), op.getType(),
                              op.getCryptoContext(), newLhs, newRhs);
    auto newKs =
        KeySwitchDownOp::create(rewriter, op.getLoc(), op.getType(),
                                op.getCryptoContext(), newOp.getResult());

    rewriter.replaceOp(op, newKs.getResult());
    if (lhsKs) rewriter.eraseOp(lhsKs);
    if (rhsKs) rewriter.eraseOp(rhsKs);

    return success();
  }
};

// Transforms: key_switch_down(key_switch_down(a)) => key_switch_down(a)
struct EliminateRedundantKeySwitchDown
    : public OpRewritePattern<KeySwitchDownOp> {
  using OpRewritePattern<KeySwitchDownOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KeySwitchDownOp op,
                                PatternRewriter& rewriter) const override {
    auto innerKs = op.getCiphertext().getDefiningOp<KeySwitchDownOp>();
    if (!innerKs) return failure();

    // replace the outer key_switch_down with the inner one
    rewriter.replaceOp(op, innerKs.getResult());
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
    patterns.add<EliminateRedundantKeySwitchDown>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
