#include "include/Dialect/TensorExt/Transforms/InsertRotate.h"

#include <utility>

#include "include/Analysis/TargetSlotAnalysis/TargetSlotAnalysis.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "insert-rotate"

namespace mlir {
namespace heir {
namespace tensor_ext {

#define GEN_PASS_DEF_INSERTROTATE
#include "include/Dialect/TensorExt/Transforms/Passes.h.inc"

namespace alignment {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "include/Dialect/TensorExt/Transforms/InsertRotate.cpp.inc"
}  // namespace alignment

namespace canonicalization {
#include "include/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"
}  // namespace canonicalization

struct InsertRotate : impl::InsertRotateBase<InsertRotate> {
  using InsertRotateBase::InsertRotateBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    // These two upstream analyses are required dependencies for any sparse
    // dataflow analysis, or else the analysis will be a no-op. Cf.
    // https://github.com/llvm/llvm-project/issues/58922
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<target_slot_analysis::TargetSlotAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    LLVM_DEBUG({
      getOperation()->walk([&](Operation *op) {
        if (op->getNumResults() == 0) return;
        auto *targetSlotLattice =
            solver.lookupState<target_slot_analysis::TargetSlotLattice>(
                op->getResult(0));
        llvm::dbgs() << "Target slot for op " << *op << ": "
                     << targetSlotLattice->getValue() << "\n";
      });
    });

    alignment::populateWithGenerated(patterns);
    canonicalization::populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
