#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/OptimizeRelinearizationAnalysis/OptimizeRelinearizationAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define DEBUG_TYPE "OptimizeRelinearization"

#define GEN_PASS_DEF_OPTIMIZERELINEARIZATION
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h.inc"

struct OptimizeRelinearization
    : impl::OptimizeRelinearizationBase<OptimizeRelinearization> {
  using OptimizeRelinearizationBase::OptimizeRelinearizationBase;

  void processSecretGenericOp(secret::GenericOp genericOp,
                              DataFlowSolver* solver) {
    // Remove all relin ops. This makes the IR invalid, because the key basis
    // sizes are incorrect. However, the correctness of the ILP ensures the key
    // basis sizes are made correct at the end.
    genericOp->walk([&](mgmt::RelinearizeOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op.erase();
    });

    OptimizeRelinearizationAnalysis analysis(
        genericOp, solver, useLocBasedVariableNames, allowMixedDegreeOperands);
    if (failed(analysis.solve())) {
      genericOp->emitError("Failed to solve the optimization problem");
      return signalPassFailure();
    }

    OpBuilder b(&getContext());

    genericOp->walk([&](Operation* op) {
      if (!analysis.shouldInsertRelin(op)) return;

      LLVM_DEBUG(llvm::dbgs()
                 << "Inserting relin after: " << op->getName() << "\n");

      b.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        auto reduceOp = mgmt::RelinearizeOp::create(b, op->getLoc(), result);
        result.replaceAllUsesExcept(reduceOp.getResult(), {reduceOp});
      }
    });
  }

  void runOnOperation() override {
    Operation* module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<DimensionAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    module->walk(
        [&](secret::GenericOp op) { processSecretGenericOp(op, &solver); });

    // optimize-relinearization will invalidate mgmt attr
    // so re-annotate it

    // temporary workaround for B/FV and all schemes of Openfhe
    auto baseLevel = 0;
    if (moduleIsBFV(getOperation()) || moduleIsOpenfhe(getOperation())) {
      // inherit mulDepth information from existing mgmt attr.
      mgmt::MgmtAttr mgmtAttr = nullptr;
      getOperation()->walk([&](secret::GenericOp op) {
        for (auto i = 0; i != op->getBlock()->getNumArguments(); ++i) {
          if ((mgmtAttr = dyn_cast<mgmt::MgmtAttr>(op.getOperandAttr(
                   i, mgmt::MgmtDialect::kArgMgmtAttrName)))) {
            break;
          }
        }
      });

      if (!mgmtAttr) {
        getOperation()->emitError(
            "No mgmt attribute found in the module for B/FV");
        return signalPassFailure();
      }

      baseLevel = mgmtAttr.getLevel();
    }

    OpPassManager pipeline("builtin.module");
    mgmt::AnnotateMgmtOptions annotateMgmtOptions;
    annotateMgmtOptions.baseLevel = baseLevel;
    pipeline.addPass(mgmt::createAnnotateMgmt(annotateMgmtOptions));
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
