#include "lib/Transforms/LowerRecip/LowerRecip.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERRECIP
#include "lib/Transforms/LowerRecip/LowerRecip.h.inc"

namespace {

Value createSplatConstant(PatternRewriter& rewriter, Location loc, Type type,
                          double value) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    auto attr = DenseElementsAttr::get(
        shaped, rewriter.getFloatAttr(shaped.getElementType(), value));
    return arith::ConstantOp::create(rewriter, loc, shaped, attr);
  }
  return arith::ConstantOp::create(
      rewriter, loc, type, cast<TypedAttr>(rewriter.getFloatAttr(type, value)));
}

struct RecipToGoldschmidt : public OpRewritePattern<math_ext::RecipOp> {
  RecipToGoldschmidt(MLIRContext* context, int numIterations)
      : OpRewritePattern<math_ext::RecipOp>(context),
        numIterations(numIterations) {}

  LogicalResult matchAndRewrite(math_ext::RecipOp op,
                                PatternRewriter& rewriter) const override {
    auto lowerAttr = op->getAttrOfType<FloatAttr>("domain_lower");
    auto upperAttr = op->getAttrOfType<FloatAttr>("domain_upper");
    if (!lowerAttr || !upperAttr) {
      return rewriter.notifyMatchFailure(
          op,
          "recip without domain_lower/domain_upper attributes has no "
          "sound Goldschmidt lowering");
    }
    double lo = lowerAttr.getValueAsDouble();
    double hi = upperAttr.getValueAsDouble();
    if (!(lo > 0.0 && hi > lo)) {
      return rewriter.notifyMatchFailure(
          op, "recip domain must satisfy 0 < domain_lower < domain_upper");
    }

    // Affine minimax seed for 1/d on [lo, hi] (equioscillating absolute
    // error of 1 - d*x0): x0(d) = alpha + beta*d with
    //   beta  = -8 / ((lo+hi)^2 + 4*lo*hi)
    //   alpha = -beta * (lo + hi)
    // Seed error E = (hi-lo)^2 / ((hi+lo)^2 + 4*lo*hi) squares per
    // Goldschmidt iteration x_{k+1} = x_k * (2 - d*x_k).
    double denom = (lo + hi) * (lo + hi) + 4 * lo * hi;
    double beta = -8.0 / denom;
    double alpha = -beta * (lo + hi);

    // Iteration count derived from the stamped domain: the seed error E
    // squares per iteration, so n iterations reach E^(2^n). Take the
    // smallest n with E^(2^n) <= kTargetError; numIterations acts as a
    // FLOOR, so tightly stamped domains keep their existing circuits
    // unchanged while wide (conservatively stamped) domains get the
    // extra iterations they need instead of losing precision. Values
    // outside the stamped domain still diverge (|1 - d*x0| > 1 beyond
    // ~hi) — the domain must cover the data; this only removes the
    // penalty for stamping it generously.
    constexpr double kTargetError = 1e-4;
    constexpr int kMaxIterations = 24;
    double seedError = (hi - lo) * (hi - lo) / denom;
    int iterations = numIterations;
    if (seedError > 0.0 && seedError < 1.0) {
      int needed = static_cast<int>(
          std::ceil(std::log2(std::log(kTargetError) / std::log(seedError))));
      iterations = std::min(std::max(iterations, needed), kMaxIterations);
    }

    Location loc = op.getLoc();
    Type type = op.getValue().getType();
    Value d = op.getValue();
    Value alphaCst = createSplatConstant(rewriter, loc, type, alpha);
    Value betaCst = createSplatConstant(rewriter, loc, type, beta);
    Value twoCst = createSplatConstant(rewriter, loc, type, 2.0);

    Value x =
        arith::AddFOp::create(rewriter, loc, alphaCst,
                              arith::MulFOp::create(rewriter, loc, betaCst, d));

    // Symmetric Goldschmidt iteration: initialize D = d * x0 and N = x0.
    // In each round, F = 2.0 - D, then D = D * F and N = N * F.
    // Both D and N are updated synchronously with the exact same factor F,
    // ensuring all multiplication operands remain at the identical level.
    Value D = arith::MulFOp::create(rewriter, loc, d, x);
    Value N = x;

    for (int i = 0; i < iterations; ++i) {
      Value F = arith::SubFOp::create(rewriter, loc, twoCst, D);
      if (i + 1 < iterations) {
        D = arith::MulFOp::create(rewriter, loc, D, F);
      }
      N = arith::MulFOp::create(rewriter, loc, N, F);
    }
    rewriter.replaceOp(op, N);
    return success();
  }

 private:
  int numIterations;
};

}  // namespace

struct LowerRecip : public impl::LowerRecipBase<LowerRecip> {
  using LowerRecipBase::LowerRecipBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RecipToGoldschmidt>(context, numIterations);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
