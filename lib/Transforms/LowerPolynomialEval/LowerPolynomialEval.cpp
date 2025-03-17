#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h"

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
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

struct LoweringBase : public OpRewritePattern<polynomial::EvalOp> {
  LoweringBase(mlir::MLIRContext *context, bool force = false,
               const std::string &dialect = "")
      : mlir::OpRewritePattern<polynomial::EvalOp>(context),
        force(force),
        dialect(dialect) {}

  Dialect *getDialect(polynomial::EvalOp op) const {
    PolynomialEvalInterface evalInterface(op.getContext());
    return dialect.empty() ? &op.getValue().getType().getDialect()
                           : op.getContext()->getOrLoadDialect(dialect);
  }

 private:
  // Force the use of this pattern, ignoring any heuristics on whether to apply
  // it.
  const bool force;

  // A dialect override provided via flag to use with the
  // PolynomialEvalInterface.
  const std::string dialect;
};

struct LowerViaHorner : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());

    // FIXME: if force, ignore heuristics on whether to use this method
    // FIXME: add lowering

    return success();
  }
};

struct LowerViaPatersonStockmeyerMonomial : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());

    // FIXME: if force, ignore heuristics on whether to use this method
    // FIXME: add lowering

    return success();
  }
};

struct LowerViaClenshaw : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());
    return failure();
  }
};

struct LowerViaPatersonStockmeyerChebyshev : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());
    return failure();
  }
};

struct LowerViaBabyStepGiantStep : public LoweringBase {
  using LoweringBase::LoweringBase;

  LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                PatternRewriter &rewriter) const override {
    Dialect *dialect = getDialect(op);
    PolynomialEvalInterface interface(op.getContext());
    return failure();
  }
};

struct LowerPolynomialEval
    : impl::LowerPolynomialEvalBase<LowerPolynomialEval> {
  using LowerPolynomialEvalBase::LowerPolynomialEvalBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if (method.hasValue() && !method.empty()) {
      if (method == "horner") {
        patterns.add<LowerViaHorner>(context, /*force=*/true, dialect);
      } else if (method == "ps") {
        patterns.add<LowerViaPatersonStockmeyerMonomial>(
            context, /*force=*/true, dialect);
      } else if (method == "clenshaw") {
        patterns.add<LowerViaClenshaw>(context, /*force=*/true, dialect);
      } else if (method == "ps-cheb") {
        patterns.add<LowerViaPatersonStockmeyerChebyshev>(
            context, /*force=*/true, dialect);
      } else if (method == "bsgs") {
        patterns.add<LowerViaBabyStepGiantStep>(context, /*force=*/true,
                                                dialect);
      } else {
        getOperation()->emitError() << "Unknown lowering method: " << method;
        signalPassFailure();
        return;
      }
    } else {
      patterns.add<LowerViaHorner, LowerViaBabyStepGiantStep,
                   LowerViaPatersonStockmeyerChebyshev, LowerViaClenshaw,
                   LowerViaPatersonStockmeyerMonomial>(context,
                                                       /*force=*/false,
                                                       dialect);
    }

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
