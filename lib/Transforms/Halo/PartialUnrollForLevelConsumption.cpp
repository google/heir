#include "lib/Transforms/Halo/PartialUnrollForLevelConsumption.h"

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Secret/IR/SecretDialect.h"
// IWYU pragma: end_keep

#define DEBUG_TYPE "partial-unroll-for-level-consumption"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PARTIALUNROLLFORLEVELCONSUMPTION
#include "lib/Transforms/Halo/Halo.h.inc"

struct PartialUnrollForLevelConsumption
    : impl::PartialUnrollForLevelConsumptionBase<
          PartialUnrollForLevelConsumption> {
  using PartialUnrollForLevelConsumptionBase::
      PartialUnrollForLevelConsumptionBase;

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

    patterns.add<PartialUnrollForLevelConsumptionAffineFor>(context, &solver);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
