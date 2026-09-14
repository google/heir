#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TensorKernelSupport.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project

namespace mlir::heir {
#define GEN_PASS_DEF_VALIDATETENSORKERNELS
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h.inc"

struct ValidateTensorKernels
    : impl::ValidateTensorKernelsBase<ValidateTensorKernels> {
  using ValidateTensorKernelsBase::ValidateTensorKernelsBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitError(
          "could not determine which tensor operands are encrypted");
      return signalPassFailure();
    }
    bool invalid = false;
    getOperation()->walk([&](Operation* op) {
      if (auto extract = dyn_cast<tensor::ExtractOp>(op)) {
        if (isSecret(extract.getTensor(), &solver))
          invalid |= failed(checkEncryptedExtract(extract));
      } else if (auto pad = dyn_cast<tensor::PadOp>(op)) {
        if (isSecret(pad.getSource(), &solver))
          invalid |= failed(checkEncryptedPad(pad));
      }
    });
    if (invalid) return signalPassFailure();
    markAllAnalysesPreserved();
  }
};
}  // namespace mlir::heir
