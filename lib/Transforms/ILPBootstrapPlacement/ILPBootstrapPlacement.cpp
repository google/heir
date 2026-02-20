#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h"

#include "lib/Analysis/ILPBootstrapPlacementAnalysis/ILPBootstrapPlacementAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ILPBOOTSTRAPPLACEMENT
#include "lib/Transforms/ILPBootstrapPlacement/ILPBootstrapPlacement.h.inc"

struct ILPBootstrapPlacement
    : impl::ILPBootstrapPlacementBase<ILPBootstrapPlacement> {
  using ILPBootstrapPlacementBase::ILPBootstrapPlacementBase;

  void processSecretGenericOp(secret::GenericOp genericOp,
                              DataFlowSolver* solver) {
    // Remove all bootstrap ops. This makes the IR invalid, because the level
    // states are incorrect. However, the correctness of the ILP ensures the
    // level states are made correct at the end.
    genericOp->walk([&](mgmt::BootstrapOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op.erase();
    });

    ILPBootstrapPlacementAnalysis analysis(genericOp, solver,
                                           bootstrapWaterline);
    if (failed(analysis.solve())) {
      genericOp->emitError(
          "Failed to solve the bootstrap placement optimization problem");
      return signalPassFailure();
    }

    // Copy bootstrap decisions before any IR changes that might invalidate
    // analysis state (e.g. inserting modreduces can reallocate the block).
    SmallVector<bool, 32> bootstrapByOpIndex =
        analysis.getBootstrapByOpIndexCopy();
    size_t solutionBootstrapCount = analysis.getSolutionBootstrapCount();
    size_t indexBootstrapCount =
        static_cast<size_t>(llvm::count(bootstrapByOpIndex, true));
    if (solutionBootstrapCount != indexBootstrapCount) {
      genericOp->emitError("ILP bootstrap placement: solution has ")
          << solutionBootstrapCount
          << " bootstraps but orderedOps index vector has "
          << indexBootstrapCount
          << " (possible solver/pointer mismatch in this environment; try "
             "--spawn_strategy=local)";
      return signalPassFailure();
    }

    OpBuilder b(&getContext());
    Block* body = genericOp.getBody();

    // Insert modreduce after every level-consuming op (e.g. mul in CKKS). The
    // ILP already solved for levels; level drops only at such ops.
    const auto isLevelConsumingOp = [](Operation& op) {
      return isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op);
    };
    DenseMap<Operation*, Operation*> levelConsumingOpToModReduce;
    for (Operation& op : body->getOperations()) {
      if (!isLevelConsumingOp(op)) continue;
      if (op.getNumResults() == 0) continue;
      Value result = op.getResult(0);
      if (!isSecret(result, solver)) continue;
      b.setInsertionPointAfter(&op);
      auto modreduce = mgmt::ModReduceOp::create(b, op.getLoc(), result);
      result.replaceAllUsesExcept(modreduce.getResult(), {modreduce});
      levelConsumingOpToModReduce.insert({&op, modreduce.getOperation()});
    }

    // Insert bootstrap at solution positions. Use op index over ops that
    // existed in the original body (analysis ran before we inserted
    // modreduce/relinearize), so skip mgmt ops we inserted.
    size_t opIndex = 0;
    for (Operation& op : body->getOperations()) {
      if (isa<secret::YieldOp>(op)) continue;
      if (isa<mgmt::ModReduceOp>(op) || isa<mgmt::RelinearizeOp>(op) ||
          isa<mgmt::BootstrapOp>(op))
        continue;
      SmallVector<Value> secretResults;
      for (OpResult result : op.getResults())
        if (isSecret(result, solver)) secretResults.push_back(result);
      if (secretResults.empty()) continue;

      bool shouldInsert =
          opIndex < bootstrapByOpIndex.size() && bootstrapByOpIndex[opIndex];
      if (!shouldInsert) {
        ++opIndex;
        continue;
      }
      ++opIndex;

      Operation* insertAfter = &op;
      SmallVector<Value> valuesToBootstrap;
      auto it = levelConsumingOpToModReduce.find(&op);
      if (it != levelConsumingOpToModReduce.end()) {
        insertAfter = it->second;
        valuesToBootstrap.push_back(insertAfter->getResult(0));
      } else {
        valuesToBootstrap.append(secretResults.begin(), secretResults.end());
      }

      for (Value result : valuesToBootstrap) {
        b.setInsertionPointAfter(insertAfter);
        auto bootstrapOp =
            mgmt::BootstrapOp::create(b, insertAfter->getLoc(), result);
        result.replaceAllUsesExcept(bootstrapOp.getResult(), {bootstrapOp});
        insertAfter = bootstrapOp.getOperation();
      }
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

    module->walk(
        [&](secret::GenericOp op) { processSecretGenericOp(op, &solver); });

    // Insert relinearization after every mul (gives mul -> relinearize ->
    // modreduce). Uses the same function as secret-insert-mgmt-ckks.
    insertRelinearizeAfterMult(getOperation(), /*includeFloats=*/true);

    // Run the same nested pipeline as secret-insert-mgmt-ckks so the pass
    // structure matches (only bootstrap placement differs: ILP vs greedy).
    // This also avoids bazel/lit stdout not reflecting our changes.
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
