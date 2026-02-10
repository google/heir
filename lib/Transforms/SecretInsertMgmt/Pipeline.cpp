#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

void runSolver(Operation* top, DataFlowSolver& solver) {
  if (failed(solver.initializeAndRun(top))) {
    LDBG() << "Failed to run solver!";
  }
}

void makeAndRunSolver(Operation* top, DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>();
  solver.load<MulDepthAnalysis>();
  runSolver(top, solver);
}

void makeAndRunSecretnessSolver(Operation* top, DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  runSolver(top, solver);
}

void makeAndRunSecretnessAndMulDepthSolver(Operation* top,
                                           DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<MulDepthAnalysis>();
  runSolver(top, solver);
}

void makeAndRunSecretnessAndLevelSolver(Operation* top,
                                        DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>();
  runSolver(top, solver);
}

LogicalResult runInsertMgmtPipeline(Operation* top,
                                    const InsertMgmtPipelineOptions& options) {
  LDBG() << "Starting insert-mgmt pipeline";
  makeLoopsTypeAndLevelInvariant(top);
  LLVM_DEBUG(top->dump());

  insertMgmtInitForPlaintexts(top, options.includeFloats);
  LLVM_DEBUG(top->dump());

  LDBG() << "Inserting mod reduce";
  insertModReduceBeforeOrAfterMult(top, options.modReduceAfterMul,
                                   options.modReduceBeforeMulIncludeFirstMul,
                                   options.includeFloats);
  LLVM_DEBUG(top->dump());

  // this must be run after ModReduceAfterMult
  LDBG() << "Inserting relinearize";
  insertRelinearizeAfterMult(top, options.includeFloats);

  LDBG() << "Unrolling loops for level consumption";
  unrollLoopsForLevelUtilization(top, options.levelBudget);
  LLVM_DEBUG(top->dump());

  // insert BootstrapOp after mgmt::ModReduceOp
  // This must be run before level mismatch
  // NOTE: actually bootstrap before mod reduce is better
  // as after modreduce to level `0` there still might be add/sub
  // and these op done there could be minimal cost.
  // However, this greedy strategy is temporary so not too much
  // optimization now
  if (options.bootstrapWaterline.has_value()) {
    LDBG() << "Bootstrap waterline";
    insertBootstrapWaterLine(top, options.bootstrapWaterline.value());
  }

  int idCounter = 0;  // for making adjust_scale op different to avoid cse
  LDBG() << "Handling cross level ops";
  handleCrossLevelOps(top, &idCounter, options.includeFloats);

  LDBG() << "Handling cross mul depth ops";
  handleCrossMulDepthOps(top, &idCounter, options.includeFloats);
  return success();
}

void insertMgmtInitForPlaintexts(Operation* top, bool includeFloats) {
  LDBG() << "Inserting mgmt.init";
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
               UseInitOpForPlaintextOperand<arith::SubIOp>,
               UseInitOpForPlaintextOperand<arith::MulIOp>,
               UseInitOpForPlaintextOperand<tensor::ExtractSliceOp>,
               UseInitOpForPlaintextOperand<tensor::InsertSliceOp>>(ctx, top,
                                                                    &solver);

  if (includeFloats) {
    patterns.add<UseInitOpForPlaintextOperand<arith::AddFOp>,
                 UseInitOpForPlaintextOperand<arith::SubFOp>,
                 UseInitOpForPlaintextOperand<arith::MulFOp>>(ctx, top,
                                                              &solver);
  }

  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertModReduceBeforeOrAfterMult(Operation* top, bool afterMul,
                                      bool beforeMulIncludeFirstMul,
                                      bool includeFloats) {
  DataFlowSolver solver;
  makeAndRunSecretnessAndMulDepthSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG({
    auto when = "before mul";
    if (afterMul) when = "after mul";
    if (beforeMulIncludeFirstMul) when = "before mul + before first mul";
    llvm::dbgs() << "Insert ModReduce " << when << "\n";
  });

  RewritePatternSet patterns(ctx);
  if (afterMul) {
    patterns.add<ModReduceAfterMult<arith::MulIOp>>(ctx, top, &solver);
    if (includeFloats)
      patterns.add<ModReduceAfterMult<arith::MulFOp>>(ctx, top, &solver);
  } else {
    patterns.add<ModReduceBefore<arith::MulIOp>>(ctx, beforeMulIncludeFirstMul,
                                                 top, &solver);
    if (includeFloats)
      patterns.add<ModReduceBefore<arith::MulFOp>>(
          ctx, beforeMulIncludeFirstMul, top, &solver);
    // includeFirstMul = false here
    // as before yield we only want mulResult to be mod reduced
    patterns.add<ModReduceBefore<secret::YieldOp>>(
        ctx, /*includeFirstMul*/ false, top, &solver);
  }
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertRelinearizeAfterMult(Operation* top, bool includeFloats) {
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MultRelinearize<arith::MulIOp>>(ctx, top, &solver);
  if (includeFloats)
    patterns.add<MultRelinearize<arith::MulFOp>>(ctx, top, &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void handleCrossLevelOps(Operation* top, int* idCounter, bool includeFloats) {
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MatchCrossLevel<arith::AddIOp>, MatchCrossLevel<arith::SubIOp>,
               MatchCrossLevel<arith::MulIOp>>(ctx, idCounter, top, &solver);
  if (includeFloats)
    patterns.add<MatchCrossLevel<arith::AddFOp>, MatchCrossLevel<arith::SubFOp>,
                 MatchCrossLevel<arith::MulFOp>>(ctx, idCounter, top, &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

// this only happen for before-mul but not include-first-mul case
// at the first level, a Value can be both mulResult or not mulResult
// we should match their scale by adding one adjust scale op
void handleCrossMulDepthOps(Operation* top, int* idCounter,
                            bool includeFloats) {
  DataFlowSolver solver;
  makeAndRunSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns
      .add<MatchCrossMulDepth<arith::AddIOp>, MatchCrossMulDepth<arith::SubIOp>,
           MatchCrossMulDepth<arith::MulIOp>>(ctx, idCounter, top, &solver);
  if (includeFloats)
    patterns.add<MatchCrossMulDepth<arith::AddFOp>,
                 MatchCrossMulDepth<arith::SubFOp>,
                 MatchCrossMulDepth<arith::MulFOp>>(ctx, idCounter, top,
                                                    &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertBootstrapWaterLine(Operation* top, int bootstrapWaterline) {
  DataFlowSolver solver;
  makeAndRunSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<BootstrapWaterLine<mgmt::ModReduceOp>>(ctx, top, &solver,
                                                      bootstrapWaterline);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void makeLoopsTypeAndLevelInvariant(Operation* top) {
  LDBG() << "Making loops type and level invariant";
  MLIRContext* ctx = top->getContext();

  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  RewritePatternSet patterns(ctx);
  patterns.add<PeelPlaintextAffineForInit, PeelPlaintextScfForInit>(ctx,
                                                                    &solver);
  walkAndApplyPatterns(top, std::move(patterns));

  DataFlowSolver solver2;
  makeAndRunSecretnessSolver(top, solver2);

  patterns.clear();
  patterns.add<BootstrapIterArgsPattern<affine::AffineForOp>,
               BootstrapIterArgsPattern<scf::ForOp>>(ctx, &solver2);
  walkAndApplyPatterns(top, std::move(patterns));
}

void unrollLoopsForLevelUtilization(Operation* top, int levelBudget) {
  DataFlowSolver solver;
  makeAndRunSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<PartialUnrollForLevelConsumptionAffineFor,
               PartialUnrollForLevelConsumptionSCFFor>(ctx, levelBudget,
                                                       &solver);
  walkAndApplyPatterns(top, std::move(patterns));

  RewritePatternSet cleanupPatterns(ctx);
  cleanupPatterns.add<DeleteAnnotatedOps>(ctx);
  walkAndApplyPatterns(top, std::move(cleanupPatterns));
}

}  // namespace heir
}  // namespace mlir
