#include "lib/Transforms/Halo/PartialUnrollForLevelConsumption.h"

#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
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

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);
    patterns.add<PartialUnrollForLevelConsumptionAffineFor,
                 PartialUnrollForLevelConsumptionSCFFor>(context, forceMaxLevel,
                                                         &solver);
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    RewritePatternSet cleanupPatterns(context);
    cleanupPatterns.add<DeleteAnnotatedOps>(context);
    walkAndApplyPatterns(getOperation(), std::move(cleanupPatterns));
  }
};

}  // namespace heir
}  // namespace mlir
