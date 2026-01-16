#include "lib/Transforms/Halo/ReconcileMixedSecretnessIterArgs.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretDialect.h"
// IWYU pragma: end_keep

#define DEBUG_TYPE "reconcile-mixed-secretness-iter-args"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RECONCILEMIXEDSECRETNESSITERARGS
#include "lib/Transforms/Halo/Halo.h.inc"

struct ReconcileMixedSecretnessIterArgs
    : impl::ReconcileMixedSecretnessIterArgsBase<
          ReconcileMixedSecretnessIterArgs> {
  using ReconcileMixedSecretnessIterArgsBase::
      ReconcileMixedSecretnessIterArgsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    patterns.add<PeelPlaintextAffineForInit, PeelPlaintextScfForInit>(context,
                                                                      &solver);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
