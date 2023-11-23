#include "include/Transforms/ValidateNoise/ValidateNoise.h"

#include "lib/Analysis/NoisePropagation/NoisePropagationAnalysis.h"
#include "lib/Analysis/NoisePropagation/Variance.h"
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                   // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_VALIDATENOISE
#include "include/Transforms/ValidateNoise/ValidateNoise.h.inc"

struct ValidateNoise : impl::ValidateNoiseBase<ValidateNoise> {
  using ValidateNoiseBase::ValidateNoiseBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    DataFlowSolver solver;
    // FIXME: do I still need DeadCodeAnalysis?
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<NoisePropagationAnalysis>();
    if (failed(solver.initializeAndRun(module))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    auto result = module->walk([&](Operation *op) {
      const VarianceLattice *opRange =
          solver.lookupState<VarianceLattice>(op->getResult(0));
      // FIXME: should be OK for some places to now know the noise.
      if (!opRange || !opRange->getValue().isKnown()) {
        op->emitOpError() << "Found op without a known noise variance; did the "
                             "analysis fail?";
        return WalkResult::interrupt();
      }

      int64_t var = opRange->getValue().getValue();
      int64_t maxNoise = 0;  // FIXME: infer from the parameters?
      if (var > maxNoise) {
        op->emitOpError() << "Found op after which the noise exceeds the "
                             "allowable maximum of "
                          << maxNoise << "; it was: " << var << "\n";
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      getOperation()->emitOpError()
          << "Detected error in the noise analysis.\n";
      signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
