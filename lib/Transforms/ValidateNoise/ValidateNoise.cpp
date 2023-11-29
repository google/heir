#include "include/Transforms/ValidateNoise/ValidateNoise.h"

#include "include/Analysis/NoisePropagation/NoisePropagationAnalysis.h"
#include "include/Analysis/NoisePropagation/Variance.h"
#include "include/Interfaces/NoiseInterfaces.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project// from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"  // from @llvm-projectject
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                   // from @llvm-project

#define DEBUG_TYPE "ValidateNoise"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_VALIDATENOISE
#include "include/Transforms/ValidateNoise/ValidateNoise.h.inc"

struct ValidateNoise : impl::ValidateNoiseBase<ValidateNoise> {
  using ValidateNoiseBase::ValidateNoiseBase;

  void runOnOperation() override {
    auto *module = getOperation();

    DataFlowSolver solver;
    // The dataflow solver needs DeadCodeAnalysis to run the other analyses
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<NoisePropagationAnalysis>();
    if (failed(solver.initializeAndRun(module))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    auto result = module->walk([&](Operation *op) {
      for (OpResult result : op->getResults()) {
        const VarianceLattice *opRange =
            solver.lookupState<VarianceLattice>(result);
        if (!opRange) {
          LLVM_DEBUG(op->emitOpError()
                     << "Solver did not assign noise to op, suggesting the "
                        "noise propagation analysis did not run properly or at "
                        "all.");
          return WalkResult::interrupt();
        }
        LLVM_DEBUG(op->emitRemark()
                   << "Found noise " << (opRange->getValue())
                   << " for op result " << result.getResultNumber());
        // It's OK for some places to not know the noise, so long as the only
        // user of that value is a bootstrap-like op.
        if (!opRange->getValue().isKnown()) {
          // One might expect a check for hasSingleUse, but there could
          // potentially be multiple downstream users, each applying a different
          // kind of programmable bootstrap to compute different functions, so
          // we loop over all users.
          for (auto result : op->getResults()) {
            for (Operation *user : result.getUsers()) {
              auto noisePropagationOp =
                  dyn_cast<NoisePropagationInterface>(user);
              // If the cast fails, then we can still proceed. The user could be
              // control flow like a func.call or a loop. In such cases, the
              // dataflow solver should propagate the value through the control
              // flow already, so we don't need to check it. It could also be a
              // decryption op, which doesn't implement the interface but is
              // valid.
              if (noisePropagationOp &&
                  !noisePropagationOp.hasArgumentIndependentResultNoise()) {
                user->emitOpError()
                    << "uses SSA value with unknown noise variance, but the op "
                       "has non-constant noise propagation. This can happen "
                       "when "
                       "an SSA value is part of control flow, such as a loop "
                       "or "
                       "an entrypoint to a function with multiple callers. In "
                       "such cases, an extra bootstrap is required to ensure "
                       "the "
                       "value does not exceed its noise bound, or the control "
                       "flow must be removed. SSA value was: \n\n"
                    << result << "\n\n";
                return WalkResult::interrupt();
              }
            }
          }

          return WalkResult::advance();
        }

        int64_t var = opRange->getValue().getValue();
        int64_t maxNoise = 0;  // FIXME: infer from the parameters?
        if (var > maxNoise) {
          op->emitOpError() << "Found op after which the noise exceeds the "
                               "allowable maximum of "
                            << maxNoise << "; it was: " << var << "\n";
          return WalkResult::interrupt();
        }
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
