#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h"

#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Utils/Approximation/CaratheodoryFejer.h"
#include "llvm/include/llvm/ADT/APFloat.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_POLYNOMIALAPPROXIMATION
#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h.inc"

using math::ExpOp;
using polynomial::EvalOp;
using polynomial::FloatPolynomial;
using polynomial::PolynomialType;
using polynomial::RingAttr;
using polynomial::TypedFloatPolynomialAttr;

APFloat exp(APFloat x) { return APFloat(std::exp(x.convertToDouble())); }

struct ConvertExp : public OpRewritePattern<ExpOp> {
  ConvertExp(mlir::MLIRContext *context)
      : OpRewritePattern<ExpOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(ExpOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    // FIXME: get degree & interval from op attribute
    FloatPolynomial poly =
        approximation::caratheodoryFejerApproximation(exp, /*degree=*/3);
    PolynomialType polyType =
        PolynomialType::get(ctx, RingAttr::get(Float64Type::get(ctx)));
    TypedFloatPolynomialAttr polyAttr =
        TypedFloatPolynomialAttr::get(polyType, poly);
    rewriter.replaceOpWithNewOp<EvalOp>(op, polyAttr, op.getOperand());
    return success();
  }
};

struct PolynomialApproximation
    : impl::PolynomialApproximationBase<PolynomialApproximation> {
  using PolynomialApproximationBase::PolynomialApproximationBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ConvertExp>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
