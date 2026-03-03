#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"

#include <cstdint>

#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"        // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"    // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ROTATIONANALYSISPASS
#include "lib/Transforms/RotationAnalysis/Passes.h.inc"

struct RotationAnalysisPass
    : impl::RotationAnalysisPassBase<RotationAnalysisPass> {
  using RotationAnalysisPassBase::RotationAnalysisPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // This analysis requires -sccp is run first to propagate constants as much
    // as possible.
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createSCCPPass());
    (void)runPipeline(pipeline, getOperation());

    RotationAnalysis analysis;
    OpBuilder builder(module);

    LogicalResult result = analysis.run(module);
    if (failed(result)) {
      llvm::errs() << "Rotation analysis failed; try re-running with "
                      "`--debug-only=rotation-analysis`"
                      " for more details";
      signalPassFailure();
      return;
    }

    auto setIndices = analysis.getRotationIndices();
    SmallVector<int64_t> indices(setIndices.begin(), setIndices.end());
    llvm::sort(indices);
    module->setAttr("rotation_analysis.indices",
                    builder.getDenseI64ArrayAttr(indices));
  }
};

}  // namespace heir
}  // namespace mlir
