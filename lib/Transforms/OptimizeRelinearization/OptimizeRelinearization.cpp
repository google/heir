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
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"           // from @llvm-project


namespace mlir {
namespace heir {

#define DEBUG_TYPE "OptimizeRelinearization"

#define GEN_PASS_DEF_OPTIMIZERELINEARIZATION
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h.inc"

struct OptimizeRelinearization
    : impl::OptimizeRelinearizationBase<OptimizeRelinearization> {
  using OptimizeRelinearizationBase::OptimizeRelinearizationBase;

  void processBlock(
      Operation* parentOp, DataFlowSolver* solver,
      const DenseMap<Operation*, SmallVector<int>>& innerLoopDegrees,
      DenseMap<Operation*, SmallVector<int>>& outLoopDegrees) {

    OptimizeRelinearizationAnalysis analysis(
        parentOp, solver, useLocBasedVariableNames, allowMixedDegreeOperands);
    
    // Pass the previously solved inner loop degrees to the outer solver
    analysis.loopBoundaryDegrees = innerLoopDegrees;

    if (failed(analysis.solve())) {
      parentOp->emitError("Failed to solve the optimization problem");
      return signalPassFailure();
    }

    OpBuilder b(&getContext());

    parentOp->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isa<LoopLikeOpInterface>(op) && op != parentOp) {
        return WalkResult::skip();
      }

      // If this is a yield op for the current loop, record its solved degrees
      // so this loop's parent can use them.
      if (isa<affine::AffineYieldOp, scf::YieldOp>(op)) {
        SmallVector<int> yieldDegrees;
        for (Value operand : op->getOperands()) {
          if (!isSecret(operand, solver)) {
            // Pad the array so the expected index lines up with the loop results
            yieldDegrees.push_back(0);
            continue;
          }
          // The output degree of the loop is the degree of the yield operand
          // after any relinearization decisions inside the loop are applied.
          int degree = analysis.keyBasisDegreeBeforeRelin(operand);
          if (auto definingOp = operand.getDefiningOp()) {
            if (analysis.shouldInsertRelin(definingOp)) {
              degree = 1;
            }
          }
          yieldDegrees.push_back(degree);
        }
        if (!yieldDegrees.empty()) {
          outLoopDegrees[parentOp] = yieldDegrees;
        }
      }

      if (!analysis.shouldInsertRelin(op)) return WalkResult::advance();

      LLVM_DEBUG(llvm::dbgs()
                 << "Inserting relin after: " << op->getName() << "\n");

      b.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        auto reduceOp = mgmt::RelinearizeOp::create(b, op->getLoc(), result);
        result.replaceAllUsesExcept(reduceOp.getResult(), {reduceOp.getOperation()});
      }
      return WalkResult::advance();
    });
  }

  void runOnOperation() override {
    Operation* module = getOperation();


    module->walk([&](secret::GenericOp genericOp) {
      genericOp->walk([&](mgmt::RelinearizeOp op) {
        op.getResult().replaceAllUsesWith(op.getOperand());
        op.erase( );
      });
    });

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<DimensionAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // Maps a loop operation to its output degrees.
    DenseMap<Operation*, SmallVector<int>> loopDegrees;

    // Process all loops bottom-up.
    module->walk<WalkOrder::PostOrder>([&](Operation* op) {
      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
        // Only process loops inside a secret.generic
        if (loopOp->getParentOfType<secret::GenericOp>()) {
          processBlock(loopOp, &solver, loopDegrees, loopDegrees);
        }
      }
    });

    // Finally, process the top-level generic ops
    module->walk([&](secret::GenericOp op) {
      processBlock(op, &solver, loopDegrees, loopDegrees);
    });

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
