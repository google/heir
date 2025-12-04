#include "lib/Transforms/AnnotateSchemeInfo/AnnotateSchemeInfo.h"

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SchemeInfoAnalysis/SchemeInfoAnalysis.h"
#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Transforms/AnnotateModule/AnnotateModule.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATESCHEMEINFO
#include "lib/Transforms/AnnotateSchemeInfo/AnnotateSchemeInfo.h.inc"

struct AnnotateSchemeInfo : impl::AnnotateSchemeInfoBase<AnnotateSchemeInfo> {
  using AnnotateSchemeInfoBase::AnnotateSchemeInfoBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<MulDepthAnalysis>();
    solver.load<SchemeInfoAnalysis>();
    solver.load<SchemeSelectionAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    annotateNatureOfComputation(getOperation(), &solver);
    auto scheme = annotateModuleWithScheme(getOperation(), &solver);

    OpPassManager pipeline("builtin.module");
    auto option = "annotate-module{scheme=" + scheme + "}";
    mlir::parsePassPipeline(option, pipeline);

    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
