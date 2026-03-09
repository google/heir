#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h"

#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#define DEBUG_TYPE "ilp-bootstrap-placement"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ILPBOOTSTRAPPLACEMENT
#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h.inc"

struct ILPBootstrapPlacement
    : impl::ILPBootstrapPlacementBase<ILPBootstrapPlacement> {
  using ILPBootstrapPlacementBase::ILPBootstrapPlacementBase;

  LogicalResult processSecretGenericOp(
      secret::GenericOp genericOp, DataFlowSolver* solver,
      SmallVector<Value, 32>* valuesToBootstrap) {
    genericOp->walk([&](mgmt::BootstrapOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op.erase();
    });

    ILPBootstrapPlacementAnalysis analysis(genericOp, solver,
                                           bootstrapWaterline);
    if (failed(analysis.solve())) {
      genericOp->emitError(
          "Failed to solve the bootstrap placement optimization problem");
      return failure();
    }
    LLVM_DEBUG(analysis.printSolution(llvm::dbgs()));
    for (Value v : analysis.getValuesToBootstrap())
      valuesToBootstrap->push_back(v);
    return success();
  }

  void insertBootstrapsForValues(ArrayRef<Value> valuesToBootstrap) {
    OpBuilder b(&getContext());
    for (Value v : valuesToBootstrap) {
      // After modreduce/relinearize we have mul -> relinearize -> modreduce.
      // Follow the chain so we bootstrap the modreduce result (correct level
      // refresh) and insert after it.
      Value toBootstrap = v;
      Operation* insertAfter = v.getDefiningOp();
      while (toBootstrap.hasOneUse()) {
        Operation* user = *toBootstrap.getUsers().begin();
        if (isa<mgmt::RelinearizeOp>(user) || isa<mgmt::ModReduceOp>(user)) {
          toBootstrap = user->getResult(0);
          insertAfter = user;
        } else {
          break;
        }
      }
      b.setInsertionPointAfter(insertAfter);
      auto bootstrapOp =
          mgmt::BootstrapOp::create(b, insertAfter->getLoc(), toBootstrap);
      toBootstrap.replaceAllUsesExcept(bootstrapOp.getResult(), {bootstrapOp});
    }
  }

  void runOnOperation() override {
    Operation* module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    SmallVector<Value, 32> valuesToBootstrap;
    auto result = module->walk([&](secret::GenericOp genericOp) {
      if (failed(
              processSecretGenericOp(genericOp, &solver, &valuesToBootstrap)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Modreduce after every mul.
    insertModReduceBeforeOrAfterMult(getOperation(), /*afterMul=*/true,
                                     /*beforeMulIncludeFirstMul=*/false,
                                     /*includeFloats=*/true);

    // Relinearize after every mul.
    insertRelinearizeAfterMult(getOperation(), /*includeFloats=*/true);

    // Insert bootstraps at the Values the ILP chose. Values remain valid.
    insertBootstrapsForValues(valuesToBootstrap);

    OpPassManager nested("builtin.module");
    nested.addPass(createCanonicalizerPass());
    nested.addPass(createCSEPass());
    nested.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(nested, getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
