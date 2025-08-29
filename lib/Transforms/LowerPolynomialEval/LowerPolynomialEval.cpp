#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h"

#include <utility>

#include "lib/Transforms/LowerPolynomialEval/Patterns.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERPOLYNOMIALEVAL
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h.inc"

struct LowerPolynomialEval
    : impl::LowerPolynomialEvalBase<LowerPolynomialEval> {
  using LowerPolynomialEvalBase::LowerPolynomialEvalBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    switch (method) {
      case PolynomialApproximationMethod::Automatic:
        patterns.add<LowerViaHorner, LowerViaPatersonStockmeyerMonomial>(
            context, /*force=*/false);
        patterns.add<LowerViaPatersonStockmeyerChebyshev>(
            context,
            /*force=*/false, minCoefficientThreshold);
        break;
      case PolynomialApproximationMethod::Horner:
        patterns.add<LowerViaHorner>(context, /*force=*/true);
        break;
      case PolynomialApproximationMethod::PatersonStockmeyer:
        patterns.add<LowerViaPatersonStockmeyerMonomial>(context,
                                                         /*force=*/true);
        break;
      case PolynomialApproximationMethod::PatersonStockmeyerChebyshev:
        patterns.add<LowerViaPatersonStockmeyerChebyshev>(
            context,
            /*force=*/true, minCoefficientThreshold);
        break;
      default:
        getOperation()->emitError() << "Unknown lowering method: " << method;
        signalPassFailure();
        return;
    }

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
