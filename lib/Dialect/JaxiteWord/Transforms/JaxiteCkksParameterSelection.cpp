#include "lib/Dialect/JaxiteWord/Transforms/JaxiteCkksParameterSelection.h"

#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxite_word {

#define GEN_PASS_DEF_JAXITECKKSPARAMETERSELECTION
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

struct JaxiteCkksParameterSelection
    : impl::JaxiteCkksParameterSelectionBase<JaxiteCkksParameterSelection> {
  using JaxiteCkksParameterSelectionBase::JaxiteCkksParameterSelectionBase;

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

}  // namespace jaxite_word
}  // namespace heir
}  // namespace mlir
