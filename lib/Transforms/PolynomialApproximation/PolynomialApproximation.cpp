#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Utils/Approximation/CaratheodoryFejer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "polynomial-approximation"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_POLYNOMIALAPPROXIMATION
#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h.inc"

constexpr int64_t kDefaultDegree = 5;
constexpr double kDefaultDomainLower = -1.0;
constexpr double kDefaultDomainUpper = 1.0;

using polynomial::EvalOp;
using polynomial::FloatPolynomial;
using polynomial::PolynomialType;
using polynomial::RingAttr;
using polynomial::TypedFloatPolynomialAttr;

inline APFloat absf(const APFloat& x) {
  return APFloat(std::abs(x.convertToDouble()));
}
inline APFloat acos(const APFloat& x) {
  return APFloat(std::acos(x.convertToDouble()));
}
inline APFloat acosh(const APFloat& x) {
  return APFloat(std::acosh(x.convertToDouble()));
}
inline APFloat asin(const APFloat& x) {
  return APFloat(std::asin(x.convertToDouble()));
}
inline APFloat asinh(const APFloat& x) {
  return APFloat(std::asinh(x.convertToDouble()));
}
inline APFloat atan(const APFloat& x) {
  return APFloat(std::atan(x.convertToDouble()));
}
inline APFloat atanh(const APFloat& x) {
  return APFloat(std::atanh(x.convertToDouble()));
}
inline APFloat cbrt(const APFloat& x) {
  return APFloat(std::cbrt(x.convertToDouble()));
}
inline APFloat ceil(const APFloat& x) {
  return APFloat(std::ceil(x.convertToDouble()));
}
inline APFloat cos(const APFloat& x) {
  return APFloat(std::cos(x.convertToDouble()));
}
inline APFloat cosh(const APFloat& x) {
  return APFloat(std::cosh(x.convertToDouble()));
}
inline APFloat erf(const APFloat& x) {
  return APFloat(std::erf(x.convertToDouble()));
}
inline APFloat erfc(const APFloat& x) {
  return APFloat(std::erfc(x.convertToDouble()));
}
inline APFloat exp(const APFloat& x) {
  return APFloat(std::exp(x.convertToDouble()));
}
inline APFloat exp2(const APFloat& x) {
  return APFloat(std::exp2(x.convertToDouble()));
}
inline APFloat expm1(const APFloat& x) {
  return APFloat(std::expm1(x.convertToDouble()));
}
inline APFloat floor(const APFloat& x) {
  return APFloat(std::floor(x.convertToDouble()));
}
inline APFloat log(const APFloat& x) {
  return APFloat(std::log(x.convertToDouble()));
}
inline APFloat log10(const APFloat& x) {
  return APFloat(std::log10(x.convertToDouble()));
}
inline APFloat log1p(const APFloat& x) {
  return APFloat(std::log1p(x.convertToDouble()));
}
inline APFloat log2(const APFloat& x) {
  return APFloat(std::log2(x.convertToDouble()));
}
inline APFloat round(const APFloat& x) {
  return APFloat(std::round(x.convertToDouble()));
}
// not available on apple cmath?
// inline APFloat _roundeven(const APFloat &x) {
//   return APFloat(roundeven(x.convertToDouble()));
// }
inline APFloat rsqrt(const APFloat& x) {
  return APFloat(1.0 / std::sqrt(x.convertToDouble()));
}
inline APFloat sin(const APFloat& x) {
  return APFloat(std::sin(x.convertToDouble()));
}
inline APFloat sinh(const APFloat& x) {
  return APFloat(std::sinh(x.convertToDouble()));
}
inline APFloat sqrt(const APFloat& x) {
  return APFloat(std::sqrt(x.convertToDouble()));
}
inline APFloat tan(const APFloat& x) {
  return APFloat(std::tan(x.convertToDouble()));
}
inline APFloat tanh(const APFloat& x) {
  return APFloat(std::tanh(x.convertToDouble()));
}
inline APFloat trunc(const APFloat& x) {
  return APFloat(std::trunc(x.convertToDouble()));
}
inline APFloat sign(const APFloat& x) {
  return APFloat(x.isNegative() ? -1.0 : (x.isZero() ? 0.0 : 1.0));
}

// Binary ops
inline APFloat atan2(const APFloat& lhs, const APFloat& rhs) {
  return APFloat(std::atan2(lhs.convertToDouble(), rhs.convertToDouble()));
}
inline APFloat fpowi(const APFloat& lhs, const APFloat& rhs) {
  return APFloat(std::pow(lhs.convertToDouble(), rhs.convertToDouble()));
}
inline APFloat powf(const APFloat& lhs, const APFloat& rhs) {
  return APFloat(std::pow(lhs.convertToDouble(), rhs.convertToDouble()));
}
inline APFloat copysign(const APFloat& lhs, const APFloat& rhs) {
  return APFloat::copySign(lhs, rhs);
}

// The user of these ops (the polynomial approximation routines) don't see the
// types of the possibly constant operand, which may be an f32 while the caller
// is using APFloats with f64 semnatics.  So we convert both operands to double
// precision and avoid this.  A better approach may be to have the polynomial
// approximation routines take as input the float semantics used to create
// APFloats internally.
inline APFloat maxf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::maximum(lhsConverted, rhsConverted);
}
inline APFloat minf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::minimum(lhsConverted, rhsConverted);
}
inline APFloat maxnumf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::maximumnum(lhsConverted, rhsConverted);
}
inline APFloat minnumf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::minimumnum(lhsConverted, rhsConverted);
}

template <typename OpTy>
struct ConvertUnaryOp : public OpRewritePattern<OpTy> {
  ConvertUnaryOp(mlir::MLIRContext* context,
                 const std::function<APFloat(APFloat)>& cppFunc)
      : OpRewritePattern<OpTy>(context, /*benefit=*/1), cppFunc(cppFunc) {}

 public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = op.getContext();
    IntegerAttr degreeAttr = op->hasAttr("degree")
                                 ? cast<IntegerAttr>(op->getAttr("degree"))
                                 : rewriter.getI32IntegerAttr(kDefaultDegree);
    FloatAttr domainLowerAttr =
        op->hasAttr("domain_lower")
            ? cast<FloatAttr>(op->getAttr("domain_lower"))
            : rewriter.getF64FloatAttr(kDefaultDomainLower);
    FloatAttr domainUpperAttr =
        op->hasAttr("domain_upper")
            ? cast<FloatAttr>(op->getAttr("domain_upper"))
            : rewriter.getF64FloatAttr(kDefaultDomainUpper);
    FloatPolynomial poly = approximation::caratheodoryFejerApproximation(
        cppFunc, degreeAttr.getInt(),
        domainLowerAttr.getValue().convertToDouble(),
        domainUpperAttr.getValue().convertToDouble());
    PolynomialType polyType =
        PolynomialType::get(ctx, RingAttr::get(Float64Type::get(ctx)));
    TypedFloatPolynomialAttr polyAttr =
        TypedFloatPolynomialAttr::get(polyType, poly);
    auto evalOp =
        rewriter.replaceOpWithNewOp<EvalOp>(op, polyAttr, op.getOperand());
    evalOp->setAttr("domain_lower", domainLowerAttr);
    evalOp->setAttr("domain_upper", domainUpperAttr);

    return success();
  }

 private:
  std::function<APFloat(APFloat)> cppFunc;
};

// Return a single value defining a constant (either a splatted tensor or a
// scalar value), or else a failure if the value is non-constant or defined by
// a non-splatted constant.
FailureOr<APFloat> getSingleValueOrSplat(Value value) {
  LLVM_DEBUG(llvm::dbgs() << "Checking if value " << value
                          << " is a constant\n");
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) {
    return failure();
  }

  auto elementsAttr = dyn_cast<ElementsAttr>(constantOp.getValue());
  if (elementsAttr && elementsAttr.isSplat()) {
    return elementsAttr.getSplatValue<APFloat>();
  }

  auto floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
  if (floatAttr) {
    return floatAttr.getValue();
  }

  auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
  if (intAttr) {
    return APFloat(APFloat::IEEEdouble(), intAttr.getValue());
  }

  return failure();
}

template <typename OpTy>
struct ConvertBinaryConstOp : public OpRewritePattern<OpTy> {
  ConvertBinaryConstOp(mlir::MLIRContext* context,
                       const std::function<APFloat(APFloat, APFloat)>& cppFunc)
      : OpRewritePattern<OpTy>(context, /*benefit=*/1), cppFunc(cppFunc) {}

 public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumOperands() != 2) {
      return op.emitOpError("Expected 2 operands; should be unreachable!");
    }

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhsConstResult = getSingleValueOrSplat(lhs);
    auto rhsConstResult = getSingleValueOrSplat(rhs);
    if (failed(lhsConstResult) && failed(rhsConstResult)) {
      // Neither operand is a single-valued constant, so we can't approximate.
      // If it's a constant but defined by a non-splatted dense elements attr,
      // we'd need to first run a pass like elementwise-to-affine to unpack the
      // tensor into individual scalars, then loop unroll or else make this pass
      // depend on SCCP analysis to get the constant here.
      return failure();
    }
    bool lhsIsConstant = succeeded(lhsConstResult);
    APFloat constValue =
        lhsIsConstant ? lhsConstResult.value() : rhsConstResult.value();
    Value nonConstOperand = lhsIsConstant ? rhs : lhs;

    // cppFunc is a binary op, so we need to give it the constant value to
    // convert it to a unary op.
    std::function<APFloat(APFloat)> unaryFunc;
    if (lhsIsConstant) {
      unaryFunc = [this, constValue](const APFloat& x) {
        return cppFunc(constValue, x);
      };
    } else {
      unaryFunc = [this, constValue](const APFloat& x) {
        return cppFunc(x, constValue);
      };
    }

    MLIRContext* ctx = op.getContext();
    IntegerAttr degreeAttr = op->hasAttr("degree")
                                 ? cast<IntegerAttr>(op->getAttr("degree"))
                                 : rewriter.getI32IntegerAttr(kDefaultDegree);
    FloatAttr domainLowerAttr =
        op->hasAttr("domain_lower")
            ? cast<FloatAttr>(op->getAttr("domain_lower"))
            : rewriter.getF64FloatAttr(kDefaultDomainLower);
    FloatAttr domainUpperAttr =
        op->hasAttr("domain_upper")
            ? cast<FloatAttr>(op->getAttr("domain_upper"))
            : rewriter.getF64FloatAttr(kDefaultDomainUpper);
    FloatPolynomial poly = approximation::caratheodoryFejerApproximation(
        unaryFunc, degreeAttr.getInt(),
        domainLowerAttr.getValue().convertToDouble(),
        domainUpperAttr.getValue().convertToDouble());
    PolynomialType polyType =
        PolynomialType::get(ctx, RingAttr::get(Float64Type::get(ctx)));
    TypedFloatPolynomialAttr polyAttr =
        TypedFloatPolynomialAttr::get(polyType, poly);
    auto evalOp =
        rewriter.replaceOpWithNewOp<EvalOp>(op, polyAttr, nonConstOperand);
    // These attributes need to be preserved when the polynomial is in the
    // Chebyshev basis, so that later passes can apply domain rescaling
    // properly.
    evalOp->setAttr("domain_lower", domainLowerAttr);
    evalOp->setAttr("domain_upper", domainUpperAttr);

    return success();
  }

 private:
  std::function<APFloat(APFloat, APFloat)> cppFunc;
};

struct PolynomialApproximation
    : impl::PolynomialApproximationBase<PolynomialApproximation> {
  using PolynomialApproximationBase::PolynomialApproximationBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
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
    patterns.add<ConvertUnaryOp<math::RsqrtOp>>(context, rsqrt);
    patterns.add<ConvertUnaryOp<math::SinOp>>(context, sin);
    patterns.add<ConvertUnaryOp<math::SinhOp>>(context, sinh);
    patterns.add<ConvertUnaryOp<math::SqrtOp>>(context, sqrt);
    patterns.add<ConvertUnaryOp<math::TanOp>>(context, tan);
    patterns.add<ConvertUnaryOp<math::TanhOp>>(context, tanh);
    patterns.add<ConvertUnaryOp<math::TruncOp>>(context, trunc);
    patterns.add<ConvertUnaryOp<math_ext::SignOp>>(context, sign);

    // TODO(#1514): Restore with alternative roundeven
    // patterns.add<ConvertUnaryOp<math::RoundEvenOp>>(context, _roundeven);

    // Unsupported math dialect unary ops:
    // math::AbsIOp
    // math::CtlzOp
    // math::CtpopOp
    // math::CttzOp
    // math::IsfiniteOp
    // math::IsinfOp
    // math::IsnanOp
    // math::IsnormalOp

    // Math binary ops (when one argument is statically constant)
    patterns.add<ConvertBinaryConstOp<arith::MaxNumFOp>>(context, maxnumf);
    patterns.add<ConvertBinaryConstOp<arith::MaximumFOp>>(context, maxf);
    patterns.add<ConvertBinaryConstOp<arith::MinNumFOp>>(context, minf);
    patterns.add<ConvertBinaryConstOp<arith::MinimumFOp>>(context, minnumf);
    patterns.add<ConvertBinaryConstOp<math::Atan2Op>>(context, atan2);
    patterns.add<ConvertBinaryConstOp<math::CopySignOp>>(context, copysign);
    patterns.add<ConvertBinaryConstOp<math::FPowIOp>>(context, fpowi);
    patterns.add<ConvertBinaryConstOp<math::PowFOp>>(context, powf);

    // Math ternary ops
    // patterns.add<ConvertUnaryOp<math::FmaOp>>(context, fma);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
