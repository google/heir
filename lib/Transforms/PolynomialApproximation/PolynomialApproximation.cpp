#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h"

#include <cmath>

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

static constexpr int64_t defaultDegree = 5;
static constexpr double defaultDomainLower = -1.0;
static constexpr double defaultDomainUpper = 1.0;

using polynomial::EvalOp;
using polynomial::FloatPolynomial;
using polynomial::PolynomialType;
using polynomial::RingAttr;
using polynomial::TypedFloatPolynomialAttr;

APFloat absf(const APFloat &x) {
  return APFloat(std::abs(x.convertToDouble()));
}
APFloat acos(const APFloat &x) {
  return APFloat(std::acos(x.convertToDouble()));
}
APFloat acosh(const APFloat &x) {
  return APFloat(std::acosh(x.convertToDouble()));
}
APFloat asin(const APFloat &x) {
  return APFloat(std::asin(x.convertToDouble()));
}
APFloat asinh(const APFloat &x) {
  return APFloat(std::asinh(x.convertToDouble()));
}
APFloat atan(const APFloat &x) {
  return APFloat(std::atan(x.convertToDouble()));
}
APFloat atanh(const APFloat &x) {
  return APFloat(std::atanh(x.convertToDouble()));
}
APFloat cbrt(const APFloat &x) {
  return APFloat(std::cbrt(x.convertToDouble()));
}
APFloat ceil(const APFloat &x) {
  return APFloat(std::ceil(x.convertToDouble()));
}
APFloat cos(const APFloat &x) { return APFloat(std::cos(x.convertToDouble())); }
APFloat cosh(const APFloat &x) {
  return APFloat(std::cosh(x.convertToDouble()));
}
APFloat erf(const APFloat &x) { return APFloat(std::erf(x.convertToDouble())); }
APFloat erfc(const APFloat &x) {
  return APFloat(std::erfc(x.convertToDouble()));
}
APFloat exp(const APFloat &x) { return APFloat(std::exp(x.convertToDouble())); }
APFloat exp2(const APFloat &x) {
  return APFloat(std::exp2(x.convertToDouble()));
}
APFloat expm1(const APFloat &x) {
  return APFloat(std::expm1(x.convertToDouble()));
}
APFloat floor(const APFloat &x) {
  return APFloat(std::floor(x.convertToDouble()));
}
APFloat log(const APFloat &x) { return APFloat(std::log(x.convertToDouble())); }
APFloat log10(const APFloat &x) {
  return APFloat(std::log10(x.convertToDouble()));
}
APFloat log1p(const APFloat &x) {
  return APFloat(std::log1p(x.convertToDouble()));
}
APFloat log2(const APFloat &x) {
  return APFloat(std::log2(x.convertToDouble()));
}
APFloat round(const APFloat &x) {
  return APFloat(std::round(x.convertToDouble()));
}
APFloat roundeven(const APFloat &x) {
  return APFloat(roundevenf(x.convertToDouble()));
}
APFloat rsqrt(const APFloat &x) {
  return APFloat(1.0 / std::sqrt(x.convertToDouble()));
}
APFloat sin(const APFloat &x) { return APFloat(std::sin(x.convertToDouble())); }
APFloat sinh(const APFloat &x) {
  return APFloat(std::sinh(x.convertToDouble()));
}
APFloat sqrt(const APFloat &x) {
  return APFloat(std::sqrt(x.convertToDouble()));
}
APFloat tan(const APFloat &x) { return APFloat(std::tan(x.convertToDouble())); }
APFloat tanh(const APFloat &x) {
  return APFloat(std::tanh(x.convertToDouble()));
}
APFloat trunc(const APFloat &x) {
  return APFloat(std::trunc(x.convertToDouble()));
}

template <typename OpTy>
struct ConvertUnaryOp : public OpRewritePattern<OpTy> {
  ConvertUnaryOp(mlir::MLIRContext *context,
                 const std::function<APFloat(APFloat)> &cppFunc)
      : OpRewritePattern<OpTy>(context, /*benefit=*/1), cppFunc(cppFunc) {}

 public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    IntegerAttr degreeAttr = op->hasAttr("degree")
                                 ? cast<IntegerAttr>(op->getAttr("degree"))
                                 : rewriter.getI32IntegerAttr(defaultDegree);
    FloatAttr domainLowerAttr =
        op->hasAttr("domain_lower")
            ? cast<FloatAttr>(op->getAttr("domain_lower"))
            : rewriter.getF64FloatAttr(defaultDomainLower);
    FloatAttr domainUpperAttr =
        op->hasAttr("domain_upper")
            ? cast<FloatAttr>(op->getAttr("domain_upper"))
            : rewriter.getF64FloatAttr(defaultDomainUpper);
    FloatPolynomial poly = approximation::caratheodoryFejerApproximation(
        exp, degreeAttr.getInt(), domainLowerAttr.getValue().convertToDouble(),
        domainUpperAttr.getValue().convertToDouble());
    PolynomialType polyType =
        PolynomialType::get(ctx, RingAttr::get(Float64Type::get(ctx)));
    TypedFloatPolynomialAttr polyAttr =
        TypedFloatPolynomialAttr::get(polyType, poly);
    rewriter.replaceOpWithNewOp<EvalOp>(op, polyAttr, op.getOperand());
    return success();
  }

 private:
  std::function<APFloat(APFloat)> cppFunc;
};

struct PolynomialApproximation
    : impl::PolynomialApproximationBase<PolynomialApproximation> {
  using PolynomialApproximationBase::PolynomialApproximationBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Math unary ops
    patterns.add<ConvertUnaryOp<math::AbsFOp>>(context, absf);
    patterns.add<ConvertUnaryOp<math::AcosOp>>(context, acos);
    patterns.add<ConvertUnaryOp<math::AcoshOp>>(context, acosh);
    patterns.add<ConvertUnaryOp<math::AsinOp>>(context, asin);
    patterns.add<ConvertUnaryOp<math::AsinhOp>>(context, asinh);
    patterns.add<ConvertUnaryOp<math::AtanOp>>(context, atan);
    patterns.add<ConvertUnaryOp<math::AtanhOp>>(context, atanh);
    patterns.add<ConvertUnaryOp<math::CbrtOp>>(context, cbrt);
    patterns.add<ConvertUnaryOp<math::CeilOp>>(context, ceil);
    patterns.add<ConvertUnaryOp<math::CosOp>>(context, cos);
    patterns.add<ConvertUnaryOp<math::CoshOp>>(context, cosh);
    patterns.add<ConvertUnaryOp<math::ErfOp>>(context, erf);
    patterns.add<ConvertUnaryOp<math::ErfcOp>>(context, erfc);
    patterns.add<ConvertUnaryOp<math::ExpOp>>(context, exp);
    patterns.add<ConvertUnaryOp<math::Exp2Op>>(context, exp2);
    patterns.add<ConvertUnaryOp<math::ExpM1Op>>(context, expm1);
    patterns.add<ConvertUnaryOp<math::FloorOp>>(context, floor);
    patterns.add<ConvertUnaryOp<math::LogOp>>(context, log);
    patterns.add<ConvertUnaryOp<math::Log10Op>>(context, log10);
    patterns.add<ConvertUnaryOp<math::Log1pOp>>(context, log1p);
    patterns.add<ConvertUnaryOp<math::Log2Op>>(context, log2);
    patterns.add<ConvertUnaryOp<math::RoundOp>>(context, round);
    patterns.add<ConvertUnaryOp<math::RoundEvenOp>>(context, roundeven);
    patterns.add<ConvertUnaryOp<math::RsqrtOp>>(context, rsqrt);
    patterns.add<ConvertUnaryOp<math::SinOp>>(context, sin);
    patterns.add<ConvertUnaryOp<math::SinhOp>>(context, sinh);
    patterns.add<ConvertUnaryOp<math::SqrtOp>>(context, sqrt);
    patterns.add<ConvertUnaryOp<math::TanOp>>(context, tan);
    patterns.add<ConvertUnaryOp<math::TanhOp>>(context, tanh);
    patterns.add<ConvertUnaryOp<math::TruncOp>>(context, trunc);

    // Unsupported math dialect unary ops:
    // math::AbsIOp
    // math::CtlzOp
    // math::CtpopOp
    // math::CttzOp
    // math::IsfiniteOp
    // math::IsinfOp
    // math::IsnanOp
    // math::IsnormalOp

    // TODO(#1487): support these ops when all but one operand is constant
    // Math binary ops (when one argument is statically constant)
    // patterns.add<ConvertUnaryOp<math::Atan2Op>>(context, atan2);
    // patterns.add<ConvertUnaryOp<math::CopySignOp>>(context, copysign);
    // patterns.add<ConvertBinaryOp<math::FpowiOp>>(context, fpowi);
    // patterns.add<ConvertBinaryOp<math::IpowiOp>>(context, ipowi);
    // patterns.add<ConvertBinaryOp<math::PowfOp>>(context, powf);

    // Math ternary ops
    // patterns.add<ConvertUnaryOp<math::FmaOp>>(context, fma);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
