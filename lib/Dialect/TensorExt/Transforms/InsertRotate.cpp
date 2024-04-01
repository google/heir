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
    LLVM_DEBUG(llvm::dbgs() << "Starting insert-rotate pass\n";);
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;

    // These two upstream analyses are required to be instantiated in any
    // sparse dataflow analysis, or else the analysis will be a no-op. Cf.
    // https://github.com/llvm/llvm-project/issues/58922
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // We want to use the result of the sparse constant propagation from the
    // first dataflow solver as an input to the target slot analysis. For some
    // reason, actually running `--sccp` before this pass causes the IR to
    // simplify away some operations that are needed to properly identify
    // target slots. So the SparseConstantPropagation above is a simulated
    // folding of arith operations, so as to identify when insertion indices
    // are statically inferable.
    //
    // TODO(#572): find a better way to depend dataflow analyses on each other.
    DataFlowSolver solver2;
    solver2.load<dataflow::DeadCodeAnalysis>();
    solver2.load<dataflow::SparseConstantPropagation>();
    solver2.load<target_slot_analysis::TargetSlotAnalysis>(symbolTable,
                                                           &solver);
    if (failed(solver2.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // Annotate all arith ops with their target slot attribute, so that it can
    // be matched in the DRR rules.
    OpBuilder builder(context);
    getOperation()->walk([&](Operation *op) {
      if (op->getNumResults() == 0) return;
      auto *targetSlotLattice =
          solver2.lookupState<target_slot_analysis::TargetSlotLattice>(
              op->getResult(0));
      if (targetSlotLattice && targetSlotLattice->getValue().isInitialized()) {
        op->setAttr(
            "target_slot",
            builder.getIndexAttr(targetSlotLattice->getValue().getValue()));
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "\nIR after attaching target slot attributes:\n"
                            << *getOperation() << "\n";);

    alignment::populateWithGenerated(patterns);
    canonicalization::populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
