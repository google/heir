#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h"

#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_POLYNOMIALAPPROXIMATION
#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h.inc"

struct PolynomialApproximation
    : impl::PolynomialApproximationBase<PolynomialApproximation> {
  using PolynomialApproximationBase::PolynomialApproximationBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // FIXME: implement pass
    patterns.add<>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
