#include "lib/Dialect/Openfhe/Transforms/CountAddAndKeySwitch.h"

#include "lib/Analysis/AddAndKeySwitchCountAnalysis/AddAndKeySwitchCountAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_COUNTADDANDKEYSWITCH
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

struct CountAddAndKeySwitch
    : impl::CountAddAndKeySwitchBase<CountAddAndKeySwitch> {
  using CountAddAndKeySwitchBase::CountAddAndKeySwitchBase;

  void runOnOperation() override {
    // skip for Lattigo backend
    // TODO(#1420): use moduleIsOpenfhe when all pipelines are ready
    if (moduleIsLattigo(getOperation())) {
      return;
    }

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    // calculate addCount/keySwitchCount
    solver.load<CountAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }
    annotateCount(getOperation(), &solver);
  }
};
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
