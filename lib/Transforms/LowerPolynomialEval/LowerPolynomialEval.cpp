#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h"

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "llvm/include/llvm/Support/Debug.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "lower-polynomial-eval"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERPOLYNOMIALEVAL
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h.inc"

using polynomial::EvalOp;
using polynomial::PolynomialType;
using polynomial::TypedFloatPolynomialAttr;
using polynomial::TypedIntPolynomialAttr;

struct LoweringBase : public OpRewritePattern<EvalOp> {
  LoweringBase(mlir::MLIRContext *context, bool force = false,
               const std::string &dialect = "")
      : mlir::OpRewritePattern<EvalOp>(context),
        force(force),
        dialect(dialect) {}

  Dialect *getDialect(EvalOp op) const {
    PolynomialEvalInterface evalInterface(op.getContext());
    return dialect.empty() ? &op.getValue().getType().getDialect()
                           : op.getContext()->getOrLoadDialect(dialect);
  }

  Attribute getAttribute(OpBuilder &b, Type type, const APInt &value) const {
    return b.getIntegerAttr(type, value);
  }

  Attribute getAttribute(OpBuilder &b, Type type, const APFloat &value) const {
    return b.getFloatAttr(type, value);
  }

  bool shouldForce() const { return force; }

 private:
  // Force the use of this pattern, ignoring any heuristics on whether to apply
  // it.
  const bool force;

  // A dialect override provided via flag to use with the
  // PolynomialEvalInterface.
  const std::string dialect;
};

template <typename PolyAttrType>
struct LowerViaHorner : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());
    Type evaluatedType = op.getValue().getType();
    auto attr = dyn_cast<PolyAttrType>(op.getPolynomialAttr());
    if (!attr) return failure();

    auto type = dyn_cast<PolynomialType>(attr.getType());
    if (!type) return failure();
    Type coeffType = type.getRing().getCoefficientType();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(op);

    auto terms = attr.getValue().getPolynomial().getTerms();
    if (terms.empty()) {
      // Handle empty polynomial case
      Value zeroConst = interface.constructConstant(
          b, op.getLoc(), b.getZeroAttr(coeffType), evaluatedType, dialect);
      rewriter.replaceOp(op, zeroConst);
      return success();
    }

    // Get the highest degree
    int64_t maxDegree = terms.back().getExponent().getSExtValue();
    const int degreeThreshold = 5;
    if (!shouldForce() && maxDegree >= degreeThreshold) return failure();

    // Create a map from exponent to coefficient for easy lookup
    std::unordered_map<int64_t, Attribute> coeffMap;
    for (auto term : terms) {
      // FIXME: construct attribute generically
      coeffMap[term.getExponent().getSExtValue()] =
          getAttribute(b, coeffType, term.getCoefficient());
    }

    // Start with the coefficient of the highest degree term
    Value result = interface.constructConstant(
        b, op.getLoc(), coeffMap[maxDegree], evaluatedType, dialect);

    // Apply Horner's method, accounting for possible missing terms
    auto x = op.getOperand();
    for (int64_t i = maxDegree - 1; i >= 0; i--) {
      // Multiply by x
      result = interface.constructMul(b, op.getLoc(), result, x, dialect);

      // Add coefficient if this term exists, otherwise continue
      if (coeffMap.find(i) != coeffMap.end()) {
        auto coeffConst = interface.constructConstant(
            b, op.getLoc(), coeffMap[i], evaluatedType, dialect);
        result =
            interface.constructAdd(b, op.getLoc(), result, coeffConst, dialect);
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename PolyAttrType>
struct LowerViaPatersonStockmeyerMonomial : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());

    // FIXME: add lowering

    return success();
  }
};

template <typename PolyAttrType>
struct LowerViaClenshaw : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());
    return failure();
  }
};

template <typename PolyAttrType>
struct LowerViaPatersonStockmeyerChebyshev : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());
    return failure();
  }
};

template <typename PolyAttrType>
struct LowerViaBabyStepGiantStep : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());
    return failure();
  }
};

using LowerViaBabyStepGiantStepFloat =
    LowerViaBabyStepGiantStep<TypedFloatPolynomialAttr>;
using LowerViaBabyStepGiantStepInt =
    LowerViaBabyStepGiantStep<TypedIntPolynomialAttr>;
using LowerViaClenshawFloat = LowerViaClenshaw<TypedFloatPolynomialAttr>;
using LowerViaClenshawInt = LowerViaClenshaw<TypedIntPolynomialAttr>;
using LowerViaHornerFloat = LowerViaHorner<TypedFloatPolynomialAttr>;
using LowerViaHornerInt = LowerViaHorner<TypedIntPolynomialAttr>;
using LowerViaPatersonStockmeyerChebyshevFloat =
    LowerViaPatersonStockmeyerChebyshev<TypedFloatPolynomialAttr>;
using LowerViaPatersonStockmeyerChebyshevInt =
    LowerViaPatersonStockmeyerChebyshev<TypedIntPolynomialAttr>;
using LowerViaPatersonStockmeyerMonomialFloat =
    LowerViaPatersonStockmeyerMonomial<TypedFloatPolynomialAttr>;
using LowerViaPatersonStockmeyerMonomialInt =
    LowerViaPatersonStockmeyerMonomial<TypedIntPolynomialAttr>;

struct LowerPolynomialEval
    : impl::LowerPolynomialEvalBase<LowerPolynomialEval> {
  using LowerPolynomialEvalBase::LowerPolynomialEvalBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if (method.hasValue() && !method.empty()) {
      if (method == "horner") {
        patterns.add<LowerViaHornerInt, LowerViaHornerFloat>(
            context, /*force=*/true, dialect);
      } else if (method == "ps") {
        patterns.add<LowerViaPatersonStockmeyerMonomialInt,
                     LowerViaPatersonStockmeyerMonomialFloat>(
            context, /*force=*/true, dialect);
      } else if (method == "clenshaw") {
        patterns.add<LowerViaClenshawInt, LowerViaClenshawFloat>(
            context, /*force=*/true, dialect);
      } else if (method == "ps-cheb") {
        patterns.add<LowerViaPatersonStockmeyerChebyshevInt,
                     LowerViaPatersonStockmeyerChebyshevFloat>(
            context, /*force=*/true, dialect);
      } else if (method == "bsgs") {
        patterns
            .add<LowerViaBabyStepGiantStepInt, LowerViaBabyStepGiantStepFloat>(
                context, /*force=*/true, dialect);
      } else {
        getOperation()->emitError() << "Unknown lowering method: " << method;
        signalPassFailure();
        return;
      }
    } else {
      patterns
          .add<LowerViaHornerInt, LowerViaHornerFloat,
               LowerViaPatersonStockmeyerMonomialInt,
               LowerViaPatersonStockmeyerMonomialFloat, LowerViaClenshawInt,
               LowerViaClenshawFloat, LowerViaPatersonStockmeyerChebyshevInt,
               LowerViaPatersonStockmeyerChebyshevFloat,
               LowerViaBabyStepGiantStepInt, LowerViaBabyStepGiantStepFloat>(
              context,
              /*force=*/false, dialect);
    }

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
