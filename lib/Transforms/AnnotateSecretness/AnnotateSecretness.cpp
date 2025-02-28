#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h"

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATESECRETNESS
#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h.inc"

struct AnnotateSecretness : impl::AnnotateSecretnessBase<AnnotateSecretness> {
  using AnnotateSecretnessBase::AnnotateSecretnessBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    annotateSecretness(getOperation(), &solver, verbose);
  }
};

}  // namespace heir
}  // namespace mlir
