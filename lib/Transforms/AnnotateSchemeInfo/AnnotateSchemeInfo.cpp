#include "lib/Transforms/AnnotateSchemeInfo/AnnotateSchemeInfo.h"

#include "lib/Analysis/SchemeInfoAnalysis/SchemeInfoAnalysis.h"
#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
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
    solver.load<SchemeInfoAnalysis>();
    solver.load<SchemeSelectionAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    annotateNatureOfComputation(getOperation(), &solver);
    annotateModuleWithScheme(getOperation(), &solver);
  }
};

}  // namespace heir
}  // namespace mlir
