#include "lib/Transforms/PolynomialApproximation/ApproximateRelu.h"

#include <utility>

#include "lib/Approximation/Sign.h"
#include "lib/Dialect/PolyExt/IR/PolyExtOps.h"
#include "lib/Dialect/Polynomial/IR/Polynomial.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_APPROXIMATERELU
#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h.inc"

using ::mlir::heir::polynomial::Monomial;
using ::mlir::heir::polynomial::Polynomial;
using ::mlir::heir::polynomial::PolynomialAttr;

class ReluFromTosaClamp : public OpRewritePattern<tosa::ClampOp> {
 public:
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp clampOp,
                                PatternRewriter& rewriter) const override {
    // ReLU is a clamp whose minimum value is 0, or the zero-point of a
    // quantized value, but the correct zero-value is not stored directly on the
    // clamp op.
    //
    // For example, we might have:
    //
    // %7 = "tosa.rescale"(%6) {
    //    double_round = true,
    //    input_zp = 0 : i32,
    //    multiplier = array<i32: 2039655736>,
    //    output_zp = -128 : i32,    <-- relevant!
    //    per_channel = false,
    //    scale32 = true,
    //    shift = array<i8: 38>
    // } : (tensor<1x16xi32>) -> tensor<1x16xi8>
    // %8 = "tosa.clamp"(%7) {
    //    max_fp = 0.000000e+00 : f32,
    //    max_int = 127 : i64,
    //    min_fp = 0.000000e+00 : f32,
    //    min_int = -128 : i64
    // } : (tensor<1x16xi8>) -> tensor<1x16xi8>
    //
    // In this case, the rescale op's output_zp must align with `min_int` to
    // be a valid ReLU op.

    auto input = clampOp.getInput();
    auto tensorTy = cast<RankedTensorType>(input.getType());
    bool isFloat = false;
    if (dyn_cast<FloatType>(tensorTy.getElementType())) {
      isFloat = true;
    }

    Attribute zeroPoint =
        TypeSwitch<Operation&, Attribute>(*input.getDefiningOp())
            .Case<tosa::RescaleOp>(
                [&](auto rescaleOp) { return rescaleOp.getOutputZpAttr(); })
            .Default([&](Operation&) {
              return isFloat ? (Attribute)rewriter.getF32FloatAttr(0)
                             : (Attribute)rewriter.getI64IntegerAttr(0);
            });

    // It's not a ReLU, but some other kind of clamping op
    if (isFloat && clampOp.getMinFpAttr() != zeroPoint) return failure();
    if (!isFloat && clampOp.getMinIntAttr() != zeroPoint) return failure();

    // At this point it's a ReLU and we can lower.
    SmallVector<Monomial> signMonomials;
    for (size_t i = 0; i < SIGN_APPROX_LEN; ++i) {
      // FIXME: only supports Integer coefficients!
      signMonomials.emplace_back(SIGN_APPROX_COEFFICIENTS[i], i);
    }
    Polynomial approximateSignPoly =
        Polynomial::fromMonomials(std::move(signMonomials), getContext());

    // Convert to max(x, 0) = (x + x sign(x)) / 2
    Monomial linearTerm(1, 1);
    Polynomial x = Polynomial::fromMonomials({linearTerm}, getContext());
    // FIXME: these are not implemented even for integer polynomials
    auto shifted = approximateSignPoly.monomialMul(1).add(x);
    auto rescaled = shifted.rescaleCoefficients(0.5);

    PolynomialAttr polyAttr = PolynomialAttr::get(approximateSignPoly);
    rewriter.replaceOpWithNewOp<poly_ext::EvalOp>(clampOp, tensorTy,
                                                  clampOp.getInput(), polyAttr);
    return success();
  }
};

struct ApproximateRelu : impl::ApproximateReluBase<ApproximateRelu> {
  using ApproximateReluBase::ApproximateReluBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReluFromTosaClamp>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
