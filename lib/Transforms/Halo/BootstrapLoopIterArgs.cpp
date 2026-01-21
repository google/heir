#include "lib/Transforms/Halo/BootstrapLoopIterArgs.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretDialect.h"
// IWYU pragma: end_keep

#define DEBUG_TYPE "bootstrap-loop-iter-args"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_BOOTSTRAPLOOPITERARGS
#include "lib/Transforms/Halo/Halo.h.inc"

struct BootstrapLoopIterArgs
    : impl::BootstrapLoopIterArgsBase<BootstrapLoopIterArgs> {
  using BootstrapLoopIterArgsBase::BootstrapLoopIterArgsBase;

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

    patterns.add<BootstrapIterArgsPattern<affine::AffineForOp>,
                 BootstrapIterArgsPattern<scf::ForOp>>(context, &solver);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
