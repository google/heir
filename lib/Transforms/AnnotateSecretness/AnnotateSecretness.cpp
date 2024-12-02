#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATESECRETNESS
#include "lib/Transforms/AnnotateSecretness/AnnotateSecretness.h.inc"

struct AnnotateSecretness : impl::AnnotateSecretnessBase<AnnotateSecretness> {
  using AnnotateSecretnessBase::AnnotateSecretnessBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

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

    // Add an attribute to the operations to show determined secretness
    OpBuilder builder(context);
    getOperation()->walk([&](Operation *op) {
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        std::string name = op->getNumResults() == 1
                               ? "secretness"
                               : "result_" + std::to_string(i) + "_secretness";
        auto *secretnessLattice =
            solver.lookupState<SecretnessLattice>(op->getOpResult(i));
        if (!secretnessLattice) {
          op->setAttr(name, builder.getStringAttr("null"));
          return;
        }
        if (!secretnessLattice->getValue().isInitialized()) {
          op->setAttr(name, builder.getStringAttr("unknown"));
          return;
        }
        op->setAttr(name, builder.getBoolAttr(
                              secretnessLattice->getValue().getSecretness()));
      }
      return;
    });
  }
};

}  // namespace heir
}  // namespace mlir
