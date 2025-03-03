#include "lib/Dialect/Openfhe/Transforms/CountAddAndKeySwitch.h"

#include "lib/Analysis/AddAndKeySwitchCountAnalysis/AddAndKeySwitchCountAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
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
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
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
