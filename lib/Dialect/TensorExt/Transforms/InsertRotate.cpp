#include "lib/Dialect/TensorExt/Transforms/InsertRotate.h"

#include <utility>

#include "lib/Analysis/TargetSlotAnalysis/TargetSlotAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_INSERTROTATE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

namespace alignment {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "lib/Dialect/TensorExt/Transforms/InsertRotate.cpp.inc"
}  // namespace alignment

namespace canonicalization {
#include "lib/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"
}  // namespace canonicalization

struct InsertRotate : impl::InsertRotateBase<InsertRotate> {
  using InsertRotateBase::InsertRotateBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<target_slot_analysis::TargetSlotAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // Annotate all arith ops with their target slot attribute, so that it can
    // be matched in the DRR rules.
    OpBuilder builder(context);
    getOperation()->walk([&](Operation* op) {
      if (op->getNumResults() == 0) return;
      auto* targetSlotLattice =
          solver.lookupState<target_slot_analysis::TargetSlotLattice>(
              op->getResult(0));
      if (targetSlotLattice && targetSlotLattice->getValue().isInitialized()) {
        op->setAttr(
            "target_slot",
            builder.getIndexAttr(targetSlotLattice->getValue().getValue()));
      }
    });

    alignment::populateWithGenerated(patterns);
    canonicalization::populateWithGenerated(patterns);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
