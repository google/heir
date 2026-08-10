#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Analysis/LevelAnalysis/BootstrapWaterlineAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallPtrSet.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/include/mlir/Rewrite/PatternApplicator.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

static bool moduleTargetsCKKS(Operation* op) {
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<ModuleOp>();
  }
  if (!module) return false;
  return moduleIsCKKS(module);
}

void runSolver(Operation* top, DataFlowSolver& solver) {
  if (failed(solver.initializeAndRun(top))) {
    LDBG() << "Failed to run solver!";
  }
}

void makeAndRunSolver(Operation* top, DataFlowSolver& solver,
                      int levelBudget = 40) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>(levelBudget);
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

void makeAndRunSecretnessAndLevelSolver(Operation* top, DataFlowSolver& solver,
                                        int levelBudget = 40) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>(levelBudget);
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

  ModuleOp module = dyn_cast<ModuleOp>(top);
  if (!module) {
    module = top->getParentOfType<ModuleOp>();
  }
  FailureOr<CompilationTarget> target = failure();
  if (module) {
    target = getTargetConfig(module);
  }

  int64_t bootstrapLevelsConsumed = 0;
  int64_t maxLevel = 40;
  if (succeeded(target)) {
    bootstrapLevelsConsumed = target->bootstrapLevelsConsumed;
  }

  int budget = options.levelBudget == -1
                   ? maxLevel
                   : (options.levelBudget + bootstrapLevelsConsumed);

  // Run Level Analysis to check for convergence
  DataFlowSolver levelSolver;
  makeAndRunSolver(top, levelSolver, budget);

  auto nonInvariantLoops = getNonInvariantLoops(top, &levelSolver);

  LDBG(2) << "Found " << nonInvariantLoops.size() << " non-invariant loops";
  for (auto* loop : nonInvariantLoops) {
    LDBG(2) << "Processing non-invariant loop " << *loop;
    DataFlowSolver secretnessSolver;
    makeAndRunSecretnessSolver(top, secretnessSolver);
    if (options.bootstrapWaterline.has_value()) {
      bootstrapLoopIterArgs(loop, &secretnessSolver);
    }

    DataFlowSolver freshLevelSolver;
    makeAndRunSolver(top, freshLevelSolver, budget);
    unrollLoopForLevelUtilization(loop, &freshLevelSolver, budget);
  }

  makeRegionBranchOpsLevelInvariant(top);

  int idCounter = 0;  // for making adjust_scale op different to avoid cse
  if (options.bootstrapWaterline.has_value()) {
    LDBG(2) << "Bootstrap waterline";
    int absoluteWaterline = budget - options.bootstrapWaterline.value();
    insertBootstrapWaterLine(top, absoluteWaterline, budget,
                             bootstrapLevelsConsumed, options.includeFloats,
                             &idCounter);
  }

  // An if statement must have each branch producing the same level as a result,
  // so the branch with the higher level must insert a level_reduce op.
  adjustLevelsForRegionBranchOps(top, budget);
  adjustScalesForRegionBranchOps(top, &idCounter);

  LDBG(2) << "Handling cross level ops";
  handleCrossLevelOps(top, &idCounter, options.includeFloats, budget);

  LDBG(2) << "Handling cross mul depth ops";
  handleCrossMulDepthOps(top, &idCounter, options.includeFloats, budget);

  // An if statement must have each branch producing the same level as a result,
  // so the branch with the higher level must insert a level_reduce op.
  adjustLevelsForRegionBranchOps(top, budget);

  // Clean up invalid modreduces and other redundant mgmt ops before final
  // analysis.
  {
    MLIRContext* ctx = top->getContext();
    RewritePatternSet cleanupPatterns(ctx);
    mgmt::ModReduceOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    mgmt::LevelReduceOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    mgmt::AdjustScaleOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    mgmt::LevelReduceMinOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    (void)walkAndApplyPatterns(top, std::move(cleanupPatterns));
  }

  DataFlowSolver finalSolver;
  makeAndRunSolver(top, finalSolver, budget);
  return validateLevelAnalysis(finalSolver, top);
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
                 UseInitOpForPlaintextOperand<arith::MulFOp>,
                 UseInitOpForPlaintextOperand<kernel::EvalChebyshevOp>>(
        ctx, top, &solver);
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
    if (includeFloats) {
      patterns.add<ModReduceBefore<arith::MulFOp>>(
          ctx, beforeMulIncludeFirstMul, top, &solver);
      patterns.add<ModReduceBefore<kernel::EvalChebyshevOp>>(
          ctx, beforeMulIncludeFirstMul, top, &solver);
    }
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

void handleCrossLevelOps(Operation* top, int* idCounter, bool includeFloats,
                         int levelBudget) {
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver, levelBudget);
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
void handleCrossMulDepthOps(Operation* top, int* idCounter, bool includeFloats,
                            int levelBudget) {
  if (moduleTargetsCKKS(top)) return;
  DataFlowSolver solver;
  makeAndRunSolver(top, solver, levelBudget);
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

static bool isSecretVal(Value val, DataFlowSolver* solver) {
  if (auto defOp = val.getDefiningOp()) {
    if (isa<mgmt::BootstrapOp>(defOp)) {
      return true;
    }
    if (isa<mgmt::AdjustScaleOp>(defOp)) {
      return isSecretVal(defOp->getOperand(0), solver);
    }
  }
  return isSecret(val, solver);
}

struct ScaleSafeTargetsResult {
  SmallVector<Value> targets;
  Operation* mulOp = nullptr;
};

static ScaleSafeTargetsResult getScaleSafeBootstrapTargets(
    Value val, DataFlowSolver* solver) {
  ScaleSafeTargetsResult result;
  Operation* defOp = val.getDefiningOp();
  if (!defOp) return result;

  if (isa<mgmt::RelinearizeOp, mgmt::AdjustScaleOp, mgmt::LevelReduceOp>(
          defOp)) {
    return getScaleSafeBootstrapTargets(defOp->getOperand(0), solver);
  }

  if (isa<arith::MulIOp, arith::MulFOp>(defOp)) {
    result.mulOp = defOp;
    for (Value operand : defOp->getOperands()) {
      if (isSecretVal(operand, solver)) {
        result.targets.push_back(operand);
      }
    }
    return result;
  }

  return result;
}

static void bootstrapMulOperands(
    Operation* mulOp, const SmallVector<Value>& scaleSafeTargets,
    OpBuilder& builder,
    llvm::DenseMap<Value, SmallVector<Value, 2>>& bootstrappedValues,
    const DominanceInfo& domInfo) {
  builder.setInsertionPoint(mulOp);
  for (unsigned i = 0; i < mulOp->getNumOperands(); ++i) {
    Value operand = mulOp->getOperand(i);
    if (llvm::is_contained(scaleSafeTargets, operand)) {
      Value bootstrappedOperand;
      if (operand.getDefiningOp() &&
          isa<mgmt::BootstrapOp>(operand.getDefiningOp())) {
        bootstrappedOperand = operand;
      } else {
        auto it = bootstrappedValues.find(operand);
        if (it != bootstrappedValues.end()) {
          for (Value cachedVal : it->second) {
            if (domInfo.dominates(cachedVal.getDefiningOp(), mulOp)) {
              bootstrappedOperand = cachedVal;
              break;
            }
          }
        }
        if (!bootstrappedOperand) {
          auto bootstrapOp = mgmt::BootstrapOp::create(
              builder, mulOp->getLoc(), operand.getType(), operand);
          bootstrappedOperand = bootstrapOp.getResult();
          bootstrappedValues[operand].push_back(bootstrappedOperand);
        }
      }
      mulOp->setOperand(i, bootstrappedOperand);
    }
  }
}

void insertBootstrapWaterLine(Operation* top, int bootstrapWaterline,
                              int levelBudget, int bootstrapLevelsConsumed,
                              bool includeFloats, int* idCounter) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<BootstrapWaterlineAnalysis>(bootstrapWaterline, levelBudget,
                                          bootstrapLevelsConsumed);
  if (failed(solver.initializeAndRun(top))) {
    LDBG() << "Failed to run solver!";
    return;
  }

  // Walk the IR to collect all Values that need bootstrapping
  SmallVector<Value> targets;
  llvm::SmallPtrSet<Value, 16> visited;
  walkValues(top, [&](Value value) {
    if (!visited.insert(value).second) {
      return;
    }
    if (isa<secret::SecretType>(value.getType())) {
      return;
    }
    if (auto* lattice = solver.lookupState<BootstrapWaterlineLattice>(value)) {
      if (lattice->getValue().getNeedsBootstrap()) {
        targets.push_back(value);
      }
    }
  });

  // Mutate the IR
  OpBuilder builder(top->getContext());
  llvm::DenseMap<Value, SmallVector<Value, 2>> bootstrappedValues;
  DominanceInfo domInfo(top);

  for (Value needsBootstrap : targets) {
    Operation* defOp = needsBootstrap.getDefiningOp();
    if (defOp && isa<ReducesLevelOpInterface>(defOp)) {
      auto reduceOp = cast<ReducesLevelOpInterface>(defOp);
      auto operandsToReduce = reduceOp.getOperandsToReduce(&solver);

      for (OpOperand* operand : operandsToReduce) {
        Value operandToReduce = operand->get();
        auto* lattice =
            solver.lookupState<BootstrapWaterlineLattice>(operandToReduce);
        if (!lattice) continue;
        LevelState level = lattice->getValue().getLevelState();
        int levelVal = 0;
        if (level.isInt()) {
          levelVal = level.getInt();
        } else if (level.isMaxLevel()) {
          levelVal = levelBudget;
        } else {
          continue;
        }
        if (!(levelVal + reduceOp.getLevelsToDrop() > bootstrapWaterline)) {
          continue;
        }

        if (moduleTargetsCKKS(top)) {
          ScaleSafeTargetsResult scaleSafeResult =
              getScaleSafeBootstrapTargets(operandToReduce, &solver);
          if (!scaleSafeResult.targets.empty()) {
            assert(scaleSafeResult.mulOp &&
                   "mulOp should not be null if scaleSafeTargets is not empty");
            bootstrapMulOperands(scaleSafeResult.mulOp, scaleSafeResult.targets,
                                 builder, bootstrappedValues, domInfo);
            continue;
          }
        }

        builder.setInsertionPoint(defOp);
        Value newOperand;
        auto it = bootstrappedValues.find(operandToReduce);
        if (it != bootstrappedValues.end()) {
          for (Value cachedVal : it->second) {
            if (domInfo.dominates(cachedVal.getDefiningOp(), defOp)) {
              newOperand = cachedVal;
              break;
            }
          }
        }
        if (!newOperand) {
          auto bootstrapOp = mgmt::BootstrapOp::create(
              builder, defOp->getLoc(), operandToReduce.getType(),
              operandToReduce);
          newOperand = bootstrapOp.getResult();
          bootstrappedValues[operandToReduce].push_back(newOperand);
        }

        if (includeFloats && isa<mgmt::ModReduceOp>(defOp) &&
            !moduleTargetsCKKS(top)) {
          auto adjustScaleOp = mgmt::AdjustScaleOp::create(
              builder, defOp->getLoc(), newOperand,
              builder.getI64IntegerAttr((*idCounter)++));
          newOperand = adjustScaleOp.getResult();
        }
        operand->set(newOperand);
      }
    } else {
      if (defOp) {
        builder.setInsertionPointAfter(defOp);
      } else {
        BlockArgument blockArg = cast<BlockArgument>(needsBootstrap);
        builder.setInsertionPointToStart(blockArg.getOwner());
      }
      Value newOperand;
      auto it = bootstrappedValues.find(needsBootstrap);
      if (it != bootstrappedValues.end()) {
        for (Value cachedVal : it->second) {
          if (defOp) {
            if (domInfo.dominates(cachedVal.getDefiningOp(), defOp)) {
              newOperand = cachedVal;
              break;
            }
          } else {
            Block* block = cast<BlockArgument>(needsBootstrap).getOwner();
            Block* cachedValBlock = cachedVal.getDefiningOp()->getBlock();
            if (cachedValBlock != block &&
                domInfo.dominates(cachedValBlock, block)) {
              newOperand = cachedVal;
              break;
            }
          }
        }
      }
      if (!newOperand) {
        auto bootstrapOp =
            mgmt::BootstrapOp::create(builder, needsBootstrap.getLoc(),
                                      needsBootstrap.getType(), needsBootstrap);
        newOperand = bootstrapOp.getResult();
        bootstrappedValues[needsBootstrap].push_back(newOperand);
      }
      needsBootstrap.replaceAllUsesExcept(newOperand,
                                          newOperand.getDefiningOp());
    }
  }
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

void adjustLevelsForRegionBranchOps(Operation* top, int levelBudget) {
  LDBG(2) << "Adjusting levels for region branching ops";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver, levelBudget);

  RewritePatternSet patterns(ctx);
  patterns.add<RegionBranchOpLevelInvariancePattern>(ctx, &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

void adjustScalesForRegionBranchOps(Operation* top, int* idCounter) {
  if (moduleTargetsCKKS(top)) return;
  LDBG(2) << "Adjusting scales for region branching ops";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessAndMulDepthSolver(top, solver);

  RewritePatternSet patterns(ctx);
  patterns.add<RegionBranchOpScaleInvariancePattern>(ctx, &solver, idCounter);
  walkAndApplyPatterns(top, std::move(patterns));
}

void unrollLoopForLevelUtilization(Operation* loopOp, DataFlowSolver* solver,
                                   int levelBudget) {
  MLIRContext* ctx = loopOp->getContext();
  PatternRewriter rewriter(ctx);
  Operation* parentOp = loopOp->getParentOp();

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
  walkAndApplyPatterns(parentOp, std::move(cleanupPatterns));
}

}  // namespace heir
}  // namespace mlir
