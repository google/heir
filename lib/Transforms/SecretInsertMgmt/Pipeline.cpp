#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"           // from @llvm-project
#include "mlir/include/mlir/Rewrite/PatternApplicator.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
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
  LDBG(2) << "Starting insert-mgmt pipeline";
  peelPlaintextIterations(top);
  LLVM_DEBUG(top->dump());

  insertMgmtInitForPlaintexts(top, options.includeFloats);
  LLVM_DEBUG(top->dump());

  LDBG(2) << "Inserting mod reduce";
  insertModReduceBeforeOrAfterMult(top, options.modReduceAfterMul,
                                   options.modReduceBeforeMulIncludeFirstMul,
                                   options.includeFloats);
  LLVM_DEBUG(top->dump());

  // this must be run after ModReduceAfterMult
  LDBG(2) << "Inserting relinearize";
  insertRelinearizeAfterMult(top, options.includeFloats);

  // Run Level Analysis to check for convergence
  DataFlowSolver levelSolver;
  makeAndRunSolver(top, levelSolver);

  auto nonInvariantLoops = getNonInvariantLoops(top, &levelSolver);

  LDBG(2) << "Found " << nonInvariantLoops.size() << " non-invariant loops";
  for (auto* loop : nonInvariantLoops) {
    LDBG(2) << "Processing non-invariant loop " << *loop;
    DataFlowSolver secretnessSolver;
    makeAndRunSecretnessSolver(top, secretnessSolver);
    bootstrapLoopIterArgs(loop, &secretnessSolver);

    DataFlowSolver freshLevelSolver;
    makeAndRunSolver(top, freshLevelSolver);
    unrollLoopForLevelUtilization(loop, &freshLevelSolver, options.levelBudget);
  }

  makeRegionBranchOpsLevelInvariant(top);

  if (options.bootstrapWaterline.has_value()) {
    LDBG(2) << "Bootstrap waterline";
    insertBootstrapWaterLine(top, options.bootstrapWaterline.value());
  }

  // An if statement must have each branch producing the same level as a result,
  // so the branch with the higher level must insert a level_reduce op.
  adjustLevelsForRegionBranchOps(top);

  int idCounter = 0;  // for making adjust_scale op different to avoid cse
  LDBG(2) << "Handling cross level ops";
  handleCrossLevelOps(top, &idCounter, options.includeFloats);

  LDBG(2) << "Handling cross mul depth ops";
  handleCrossMulDepthOps(top, &idCounter, options.includeFloats);

  // An if statement must have each branch producing the same level as a result,
  // so the branch with the higher level must insert a level_reduce op.
  adjustLevelsForRegionBranchOps(top);
  return success();
}

void insertMgmtInitForPlaintexts(Operation* top, bool includeFloats) {
  LDBG(2) << "Inserting mgmt.init";
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
               UseInitOpForPlaintextOperand<arith::SubIOp>,
               UseInitOpForPlaintextOperand<arith::MulIOp>,
               UseInitOpForPlaintextOperand<tensor::ExtractSliceOp>,
               UseInitOpForPlaintextOperand<tensor::InsertSliceOp>,
               UseInitOpForPlaintextOperand<tensor::InsertOp>>(ctx, top,
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
               MatchCrossLevel<arith::MulIOp>,
               MatchCrossLevel<tensor::InsertSliceOp>,
               MatchCrossLevel<tensor::InsertOp>>(ctx, idCounter, top, &solver);
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
           MatchCrossMulDepth<arith::MulIOp>,
           MatchCrossMulDepth<tensor::InsertSliceOp>,
           MatchCrossMulDepth<tensor::InsertOp>>(ctx, idCounter, top, &solver);
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

void peelPlaintextIterations(Operation* top) {
  LDBG(2) << "Peeling plaintext iterations";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);
  RewritePatternSet patterns(ctx);
  patterns.add<PeelPlaintextAffineForInit, PeelPlaintextScfForInit>(ctx,
                                                                    &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

void bootstrapLoopIterArgs(Operation* loopOp, DataFlowSolver* solver) {
  LDBG(2) << "Bootstrapping loop iter args";
  MLIRContext* ctx = loopOp->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<BootstrapIterArgsPattern<affine::AffineForOp>,
               BootstrapIterArgsPattern<scf::ForOp>>(ctx, solver);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  PatternApplicator applicator(frozenPatterns);
  applicator.applyDefaultCostModel();

  PatternRewriter rewriter(ctx);
  (void)applicator.matchAndRewrite(loopOp, rewriter);
}

void makeRegionBranchOpsLevelInvariant(Operation* top) {
  LDBG(2) << "Making region branch ops level invariant";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);
  RewritePatternSet patterns(ctx);
  patterns.add<UseInitForPlaintextBranchTerminators,
               RegionBranchOpLevelInvariancePattern>(ctx, &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

SmallVector<Operation*> getNonInvariantLoops(Operation* top,
                                             DataFlowSolver* solver) {
  LDBG(2) << "Getting non-invariant loops";
  SmallVector<Operation*> nonInvariantLoops;

  auto isInvariant = [&](LoopLikeOpInterface forOp) {
    for (auto [i, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (!isSecret(iterArg, solver)) continue;

      auto* initLattice =
          solver->lookupState<LevelLattice>(forOp.getInits()[i]);
      auto yieldedValue =
          forOp.getTiedLoopYieldedValue(cast<BlockArgument>(iterArg))->get();
      auto* yieldLattice = solver->lookupState<LevelLattice>(yieldedValue);

      if (!initLattice || !yieldLattice || !initLattice->getValue().isInt() ||
          !yieldLattice->getValue().isInt()) {
        return false;
      }

      if (initLattice->getValue().getInt() !=
          yieldLattice->getValue().getInt()) {
        return false;
      }
    }
    return true;
  };

  // Post-order walk means we process nested loops from the inside out.
  top->walk<WalkOrder::PostOrder>([&](Operation* op) {
    if (isa<affine::AffineForOp, scf::ForOp>(op)) {
      if (!isInvariant(cast<LoopLikeOpInterface>(op))) {
        nonInvariantLoops.push_back(op);
      }
    }
  });

  return nonInvariantLoops;
}

void adjustLevelsForRegionBranchOps(Operation* top) {
  LDBG(2) << "Adjusting levels for region branching ops";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver);

  RewritePatternSet patterns(ctx);
  patterns.add<RegionBranchOpLevelInvariancePattern>(ctx, &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

void unrollLoopForLevelUtilization(Operation* loopOp, DataFlowSolver* solver,
                                   int levelBudget) {
  MLIRContext* ctx = loopOp->getContext();
  PatternRewriter rewriter(ctx);

  // A pattern driver is not appropriate here because we need to unroll
  // the loops from inner-most to outer-most. The order in which nested
  // loops are returned from getNonInvariantLoops ensures this.
  TypeSwitch<Operation*>(loopOp)
      .Case<affine::AffineForOp, scf::ForOp>([&](auto op) {
        (void)doPartialUnroll(op, rewriter, levelBudget, solver);
      })
      .Default([&](auto op) {
        LDBG(2) << "Unknown loop type " << loopOp->getName();
      });

  LDBG(2) << "Deleting annotated ops";
  RewritePatternSet cleanupPatterns(ctx);
  cleanupPatterns.add<DeleteAnnotatedOps>(ctx);
  walkAndApplyPatterns(loopOp, std::move(cleanupPatterns));
}

}  // namespace heir
}  // namespace mlir
