#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

#include <algorithm>
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
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallPtrSet.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
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
  // A default large value when the user does not set a level budget. Used for
  // ensuring subesquent passes that require a level budget get one.
  int64_t maxLevel = 100;
  if (succeeded(target)) {
    bootstrapLevelsConsumed = target->bootstrapLevelsConsumed;
  }

  int budget = options.levelBudget == -1
                   ? maxLevel
                   : (options.levelBudget + bootstrapLevelsConsumed);

  int idCounter = 0;  // for making adjust_scale op different to avoid cse

  // 1. Hoist invariant bootstraps first
  if (options.bootstrapWaterline.has_value()) {
    LDBG(2) << "Running insertBootstrapWaterLine (only hoist) first";
    int absoluteWaterline = budget - options.bootstrapWaterline.value();
    insertBootstrapWaterLine(top, absoluteWaterline, budget,
                             bootstrapLevelsConsumed, options.includeFloats,
                             &idCounter, /*onlyHoist=*/true);
  }

  // 2. Re-solve levels to propagate hoisted levels
  int solverBudget = std::max(budget, (int)maxLevel);
  DataFlowSolver levelSolver;
  makeAndRunSolver(top, levelSolver, solverBudget);

  // 3. Make region branch ops level invariant (using updated levels)
  makeRegionBranchOpsLevelInvariant(top, solverBudget);

  // 4. Re-solve levels again to propagate aligned levels
  DataFlowSolver levelSolver2;
  makeAndRunSolver(top, levelSolver2, solverBudget);

  // 5. Get non-invariant loops now
  auto nonInvariantLoops = getNonInvariantLoops(top, &levelSolver2);
  LDBG(2) << "Found " << nonInvariantLoops.size() << " non-invariant loops";

  // 6. Bootstrap loop iter args for the remaining non-invariant loops
  if (options.bootstrapWaterline.has_value()) {
    for (auto* loop : nonInvariantLoops) {
      LDBG(2) << "Bootstrapping iter args for non-invariant loop " << *loop;
      DataFlowSolver secretnessSolver;
      makeAndRunSecretnessSolver(top, secretnessSolver);
      bootstrapLoopIterArgs(loop, &secretnessSolver);
    }
  }

  // 7. Unroll remaining non-invariant loops
  for (auto* loop : nonInvariantLoops) {
    LDBG(2) << "Unrolling non-invariant loop " << *loop;
    DataFlowSolver freshLevelSolver;
    makeAndRunSolver(top, freshLevelSolver, budget);
    unrollLoopForLevelUtilization(loop, &freshLevelSolver, budget);
  }

  // 8. Run full waterline bootstrap insertion
  if (options.bootstrapWaterline.has_value()) {
    LDBG(2) << "Running insertBootstrapWaterLine (full) after unrolling";
    int absoluteWaterline = budget - options.bootstrapWaterline.value();
    insertBootstrapWaterLine(top, absoluteWaterline, budget,
                             bootstrapLevelsConsumed, options.includeFloats,
                             &idCounter, /*onlyHoist=*/false);
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

static LoopLikeOpInterface getOutermostLoopForInvariant(Value val,
                                                        Operation* userOp) {
  LoopLikeOpInterface outermostLoop = nullptr;
  Operation* curr = userOp->getParentOp();
  while (curr) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(curr)) {
      if (loop.isDefinedOutsideOfLoop(val)) {
        outermostLoop = loop;
      } else {
        break;
      }
    }
    curr = curr->getParentOp();
  }
  return outermostLoop;
}

static Value bootstrapValue(
    Value val, OpBuilder& builder,
    llvm::DenseMap<Value, SmallVector<Value, 2>>& bootstrappedValues,
    const DominanceInfo& domInfo, Operation* insertionPoint = nullptr) {
  if (val.getDefiningOp() && isa<mgmt::BootstrapOp>(val.getDefiningOp())) {
    return val;
  }
  Operation* defOp = val.getDefiningOp();
  Value newOperand;
  {
    OpBuilder::InsertionGuard guard(builder);
    if (insertionPoint) {
      builder.setInsertionPoint(insertionPoint);
    } else if (defOp) {
      builder.setInsertionPointAfter(defOp);
    } else {
      BlockArgument blockArg = cast<BlockArgument>(val);
      builder.setInsertionPointToStart(blockArg.getOwner());
    }

    Operation* actualInsertionPoint = insertionPoint;
    if (!actualInsertionPoint) {
      if (defOp) {
        actualInsertionPoint = defOp;
      }
    }

    auto it = bootstrappedValues.find(val);
    if (it != bootstrappedValues.end()) {
      for (Value cachedVal : it->second) {
        if (actualInsertionPoint) {
          if (domInfo.dominates(cachedVal.getDefiningOp(),
                                actualInsertionPoint)) {
            newOperand = cachedVal;
            break;
          }
        } else {
          Block* block = cast<BlockArgument>(val).getOwner();
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
      Location loc = insertionPoint ? insertionPoint->getLoc() : val.getLoc();
      auto bootstrapOp =
          mgmt::BootstrapOp::create(builder, loc, val.getType(), val);
      newOperand = bootstrapOp.getResult();
      bootstrappedValues[val].push_back(newOperand);
    }
  }
  return newOperand;
}

static bool isRotationByZero(Value val, Value source) {
  auto rotateOp = val.getDefiningOp<tensor_ext::RotateOp>();
  if (!rotateOp) return false;
  if (rotateOp.getOperand(0) != source) return false;
  auto constOp = rotateOp.getOperand(1).getDefiningOp<arith::ConstantOp>();
  if (!constOp) return false;
  auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr) return false;
  return intAttr.getInt() == 0;
}

static Value getSourceOfRotations(Value val) {
  Operation* defOp = val.getDefiningOp();
  if (!defOp) return nullptr;

  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    if (forOp.getNumResults() != 1 ||
        !isa<TensorType>(forOp.getResult(0).getType())) {
      return nullptr;
    }
    Block* body = forOp.getBody();
    auto yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) return nullptr;
    Value yieldedVal = yieldOp.getOperand(0);

    auto insertSliceOp = yieldedVal.getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) return nullptr;

    Value insertedVal = insertSliceOp.getSource();
    auto rotateOp = insertedVal.getDefiningOp<tensor_ext::RotateOp>();
    if (!rotateOp) return nullptr;

    Value rotateSource = rotateOp.getOperand(0);
    if (!forOp.isDefinedOutsideOfLoop(rotateSource)) {
      return nullptr;
    }

    Value initVal = forOp.getInitArgs()[0];
    auto initInsertSliceOp = initVal.getDefiningOp<tensor::InsertSliceOp>();
    if (initInsertSliceOp) {
      Value initSource = initInsertSliceOp.getSource();
      if (initSource == rotateSource ||
          isRotationByZero(initSource, rotateSource)) {
        return rotateSource;
      }
    }
  }
  return nullptr;
}

// Modify the rotations loop in-place to use bootstrappedRotSource
static void useRotSource(scf::ForOp forOp,
                         tensor::InsertSliceOp initInsertSliceOp,
                         Value bootstrappedRotSource) {
  Block* body = forOp.getBody();
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  auto insertSliceOp =
      cast<tensor::InsertSliceOp>(yieldOp.getOperand(0).getDefiningOp());
  auto rotateOp =
      cast<tensor_ext::RotateOp>(insertSliceOp.getSource().getDefiningOp());
  rotateOp.setOperand(0, bootstrappedRotSource);
  initInsertSliceOp.setOperand(0, bootstrappedRotSource);
}

static bool tryHoistBootstrap(
    Value val, OpBuilder& builder,
    llvm::DenseMap<Value, SmallVector<Value, 2>>& hoistedTensors,
    const DominanceInfo& domInfo) {
  Operation* valDefOp = val.getDefiningOp();
  if (valDefOp && (isa<tensor::ExtractSliceOp>(valDefOp) ||
                   isa<tensor::ExtractOp>(valDefOp))) {
    Value source = valDefOp->getOperand(0);
    if (source.getDefiningOp() &&
        isa<mgmt::BootstrapOp>(source.getDefiningOp())) {
      return true;
    }

    // Check if source is a rotations loop
    if (Value rotSource = getSourceOfRotations(source)) {
      if (rotSource.getDefiningOp() &&
          isa<mgmt::BootstrapOp>(rotSource.getDefiningOp())) {
        // Already bootstrapped
        return true;
      }

      Operation* rotationsLoopOp = source.getDefiningOp();
      auto forOp = cast<scf::ForOp>(rotationsLoopOp);
      Value initVal = forOp.getInitArgs()[0];
      auto initInsertSliceOp =
          cast<tensor::InsertSliceOp>(initVal.getDefiningOp());

      // Bootstrap the source of rotations before the initializer of the loop
      Value bootstrappedRotSource;
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(initInsertSliceOp);
        Location loc = rotSource.getLoc();
        auto bootstrapOp = mgmt::BootstrapOp::create(
            builder, loc, rotSource.getType(), rotSource);
        bootstrappedRotSource = bootstrapOp.getResult();
      }

      useRotSource(forOp, initInsertSliceOp, bootstrappedRotSource);

      return true;
    }

    // 1. Try to reuse cached bootstrap
    auto it = hoistedTensors.find(source);
    if (it != hoistedTensors.end()) {
      for (Value cachedVal : it->second) {
        if (domInfo.dominates(cachedVal.getDefiningOp(), valDefOp)) {
          valDefOp->setOperand(0, cachedVal);
          return true;
        }
      }
    }

    // 2. Try to hoist
    if (LoopLikeOpInterface outermostLoop =
            getOutermostLoopForInvariant(source, valDefOp)) {
      Value bootstrappedSource;
      OpBuilder::InsertionGuard guard(builder);
      if (Operation* sourceDefOp = source.getDefiningOp()) {
        if (sourceDefOp->getBlock() == outermostLoop->getBlock()) {
          builder.setInsertionPointAfter(sourceDefOp);
        } else {
          builder.setInsertionPoint(outermostLoop);
        }
      } else {
        BlockArgument blockArg = cast<BlockArgument>(source);
        if (blockArg.getOwner() == outermostLoop->getBlock()) {
          builder.setInsertionPointToStart(blockArg.getOwner());
        } else {
          builder.setInsertionPoint(outermostLoop);
        }
      }

      Location loc = source.getLoc();
      auto bootstrapOp =
          mgmt::BootstrapOp::create(builder, loc, source.getType(), source);
      bootstrappedSource = bootstrapOp.getResult();
      hoistedTensors[source].push_back(bootstrappedSource);

      // Replace all uses of source within the outermostLoop with
      // bootstrappedSource
      source.replaceUsesWithIf(bootstrappedSource, [&](OpOperand& operand) {
        return outermostLoop.getOperation()->isAncestor(operand.getOwner());
      });
      return true;
    }
  }
  return false;
}

static void bootstrapMulOperands(
    Operation* mulOp, const SmallVector<Value>& scaleSafeTargets,
    OpBuilder& builder,
    llvm::DenseMap<Value, SmallVector<Value, 2>>& bootstrappedValues,
    const DominanceInfo& domInfo, bool replaceDominatedUses = false) {
  for (unsigned i = 0; i < mulOp->getNumOperands(); ++i) {
    Value operand = mulOp->getOperand(i);
    if (llvm::is_contained(scaleSafeTargets, operand)) {
      Operation* insertionPoint = mulOp;
      if (auto loop = getOutermostLoopForInvariant(operand, mulOp)) {
        insertionPoint = loop;
      }
      Value bootstrappedOperand = bootstrapValue(
          operand, builder, bootstrappedValues, domInfo, insertionPoint);
      if (replaceDominatedUses) {
        Operation* bootstrapOp = bootstrappedOperand.getDefiningOp();
        operand.replaceUsesWithIf(bootstrappedOperand, [&](OpOperand& use) {
          return use.getOwner() != bootstrapOp &&
                 domInfo.dominates(bootstrapOp, use.getOwner());
        });
      } else {
        mulOp->setOperand(i, bootstrappedOperand);
      }
    }
  }
}

static void insertCKKSMulHeadroomBootstraps(Operation* top,
                                            int bootstrapWaterline,
                                            int levelBudget,
                                            int bootstrapLevelsConsumed) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<BootstrapWaterlineAnalysis>(bootstrapWaterline, levelBudget,
                                          bootstrapLevelsConsumed,
                                          /*requireMulHeadroom=*/true);
  if (failed(solver.initializeAndRun(top))) {
    LDBG() << "Failed to run multiplication headroom analysis!";
    return;
  }

  SmallVector<ScaleSafeTargetsResult> targets;
  top->walk([&](Operation* op) {
    if (!isa<IncreasesMulDepthOpInterface>(op) || op->getNumResults() != 1) {
      return;
    }
    auto* lattice =
        solver.lookupState<BootstrapWaterlineLattice>(op->getResult(0));
    if (!lattice || !lattice->getValue().getNeedsBootstrap()) {
      return;
    }
    ScaleSafeTargetsResult target =
        getScaleSafeBootstrapTargets(op->getResult(0), &solver);
    if (!target.targets.empty()) {
      targets.push_back(std::move(target));
    }
  });

  OpBuilder builder(top->getContext());
  llvm::DenseMap<Value, SmallVector<Value, 2>> bootstrappedValues;
  DominanceInfo domInfo(top);
  for (const ScaleSafeTargetsResult& target : targets) {
    bootstrapMulOperands(target.mulOp, target.targets, builder,
                         bootstrappedValues, domInfo,
                         /*replaceDominatedUses=*/true);
  }
}

void insertBootstrapWaterLine(Operation* top, int bootstrapWaterline,
                              int levelBudget, int bootstrapLevelsConsumed,
                              bool includeFloats, int* idCounter,
                              bool onlyHoist) {
  if (!onlyHoist && moduleTargetsCKKS(top)) {
    insertCKKSMulHeadroomBootstraps(top, bootstrapWaterline, levelBudget,
                                    bootstrapLevelsConsumed);
  }

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
    // Only a ciphertext can be bootstrapped. A plaintext has no level state
    // for the analysis to track, which reads as "needs bootstrap", and
    // refreshing it would leave a mgmt.bootstrap on a plaintext operand that
    // no lowering can convert.
    if (!isSecret(value, &solver)) {
      return;
    }
    if (auto* lattice = solver.lookupState<BootstrapWaterlineLattice>(value)) {
      if (lattice->getValue().getNeedsBootstrap()) {
        targets.push_back(value);
      }
    }
  });

  OpBuilder builder(top->getContext());
  llvm::DenseMap<Value, SmallVector<Value, 2>> bootstrappedValues;
  DominanceInfo domInfo(top);
  llvm::DenseMap<Value, SmallVector<Value, 2>> hoistedTensors;

  // Phase 1: Identify and insert hoisted bootstraps
  for (Value needsBootstrap : targets) {
    Operation* defOp = needsBootstrap.getDefiningOp();
    SmallVector<Value> leafTargets;
    if (defOp && isa<ReducesLevelOpInterface>(defOp)) {
      auto reduceOp = cast<ReducesLevelOpInterface>(defOp);
      auto operandsToReduce = reduceOp.getOperandsToReduce(&solver);
      for (OpOperand* operand : operandsToReduce) {
        Value operandToReduce = operand->get();
        if (moduleTargetsCKKS(top)) {
          ScaleSafeTargetsResult scaleSafeResult =
              getScaleSafeBootstrapTargets(operandToReduce, &solver);
          if (!scaleSafeResult.targets.empty()) {
            leafTargets.append(scaleSafeResult.targets.begin(),
                               scaleSafeResult.targets.end());
          } else {
            leafTargets.push_back(operandToReduce);
          }
        } else {
          leafTargets.push_back(operandToReduce);
        }
      }
    } else {
      leafTargets.push_back(needsBootstrap);
    }

    for (Value leaf : leafTargets) {
      Operation* leafDefOp = leaf.getDefiningOp();
      if (leafDefOp && (isa<tensor::ExtractSliceOp>(leafDefOp) ||
                        isa<tensor::ExtractOp>(leafDefOp))) {
        Value source = leafDefOp->getOperand(0);
        if (source.getDefiningOp() &&
            isa<mgmt::BootstrapOp>(source.getDefiningOp())) {
          continue;
        }

        // Check if source is a rotations loop
        if (Value rotSource = getSourceOfRotations(source)) {
          if (rotSource.getDefiningOp() &&
              isa<mgmt::BootstrapOp>(rotSource.getDefiningOp())) {
            continue;
          }

          Operation* rotationsLoopOp = source.getDefiningOp();
          auto forOp = cast<scf::ForOp>(rotationsLoopOp);
          Value initVal = forOp.getInitArgs()[0];
          auto initInsertSliceOp =
              cast<tensor::InsertSliceOp>(initVal.getDefiningOp());

          // Bootstrap the source of rotations before the initializer of the
          // loop
          Value bootstrappedRotSource;
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(initInsertSliceOp);
            Location loc = rotSource.getLoc();
            auto bootstrapOp = mgmt::BootstrapOp::create(
                builder, loc, rotSource.getType(), rotSource);
            bootstrappedRotSource = bootstrapOp.getResult();
          }

          useRotSource(forOp, initInsertSliceOp, bootstrappedRotSource);
          continue;
        }

        if (hoistedTensors.count(source)) {
          continue;
        }
        if (LoopLikeOpInterface outermostLoop =
                getOutermostLoopForInvariant(source, leafDefOp)) {
          Value bootstrappedSource;
          OpBuilder::InsertionGuard guard(builder);
          if (Operation* sourceDefOp = source.getDefiningOp()) {
            if (sourceDefOp->getBlock() == outermostLoop->getBlock()) {
              builder.setInsertionPointAfter(sourceDefOp);
            } else {
              builder.setInsertionPoint(outermostLoop);
            }
          } else {
            BlockArgument blockArg = cast<BlockArgument>(source);
            if (blockArg.getOwner() == outermostLoop->getBlock()) {
              builder.setInsertionPointToStart(blockArg.getOwner());
            } else {
              builder.setInsertionPoint(outermostLoop);
            }
          }

          Location loc = source.getLoc();
          auto bootstrapOp =
              mgmt::BootstrapOp::create(builder, loc, source.getType(), source);
          bootstrappedSource = bootstrapOp.getResult();
          hoistedTensors[source].push_back(bootstrappedSource);
        }
      }
    }
  }

  // Phase 2: Perform replacements using the cache
  top->walk([&](Operation* op) {
    if (isa<tensor::ExtractSliceOp>(op) || isa<tensor::ExtractOp>(op)) {
      Value source = op->getOperand(0);
      auto it = hoistedTensors.find(source);
      if (it != hoistedTensors.end()) {
        for (Value cachedVal : it->second) {
          if (domInfo.dominates(cachedVal.getDefiningOp(), op)) {
            op->setOperand(0, cachedVal);
            break;
          }
        }
      }
    }
  });

  if (onlyHoist) {
    return;
  }

  // Phase 3: Insert remaining (non-hoisted) bootstraps
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
            SmallVector<Value> remainingTargets;
            for (Value target : scaleSafeResult.targets) {
              if (!tryHoistBootstrap(target, builder, hoistedTensors,
                                     domInfo)) {
                remainingTargets.push_back(target);
              }
            }
            if (!remainingTargets.empty()) {
              bootstrapMulOperands(scaleSafeResult.mulOp, remainingTargets,
                                   builder, bootstrappedValues, domInfo);
            }
            continue;
          }
        }

        if (tryHoistBootstrap(operandToReduce, builder, hoistedTensors,
                              domInfo)) {
          continue;
        }

        Operation* insertionPoint = defOp;
        if (auto loop = getOutermostLoopForInvariant(operandToReduce, defOp)) {
          insertionPoint = loop;
        }

        Value newOperand =
            bootstrapValue(operandToReduce, builder, bootstrappedValues,
                           domInfo, insertionPoint);

        if (includeFloats && isa<mgmt::ModReduceOp>(defOp) &&
            !moduleTargetsCKKS(top)) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(insertionPoint);
          auto adjustScaleOp = mgmt::AdjustScaleOp::create(
              builder, defOp->getLoc(), newOperand,
              builder.getI64IntegerAttr((*idCounter)++));
          newOperand = adjustScaleOp.getResult();
        }
        operand->set(newOperand);
      }
    } else {
      if (tryHoistBootstrap(needsBootstrap, builder, hoistedTensors, domInfo)) {
        continue;
      }
      Value newOperand =
          bootstrapValue(needsBootstrap, builder, bootstrappedValues, domInfo);
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

void makeRegionBranchOpsLevelInvariant(Operation* top, int levelBudget) {
  LDBG(2) << "Making region branch ops level invariant";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver, levelBudget);
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
    LDBG(2) << "Checking invariance for loop: " << *forOp;
    for (auto [i, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (!isSecret(iterArg, solver)) {
        continue;
      }

      auto* initLattice =
          solver->lookupState<LevelLattice>(forOp.getInits()[i]);
      auto yieldedValue =
          forOp.getTiedLoopYieldedValue(cast<BlockArgument>(iterArg))->get();
      auto* yieldLattice = solver->lookupState<LevelLattice>(yieldedValue);

      if (!initLattice || !yieldLattice) {
        LDBG(2) << "  Arg " << i << ": missing lattice -> non-invariant";
        return false;
      }

      LevelState initVal = initLattice->getValue();
      LevelState yieldVal = yieldLattice->getValue();

      LDBG(2) << "  Arg " << i << ": initVal=" << initVal
              << ", yieldVal=" << yieldVal;

      if (!initVal.isInitialized() && yieldVal.isInitialized()) {
        // Init is Uninit, yield is concrete. We can align init to yield.
        // Treat as invariant.
        LDBG(2) << "  Arg " << i << ": treated as invariant (Uninit init)";
        continue;
      }

      if (!initVal.isInt() || !yieldVal.isInt()) {
        LDBG(2) << "  Arg " << i << ": not int -> non-invariant";
        return false;
      }

      if (initVal.getInt() != yieldVal.getInt()) {
        LDBG(2) << "  Arg " << i << ": levels mismatch (" << initVal.getInt()
                << " != " << yieldVal.getInt() << ") -> non-invariant";
        return false;
      }
    }
    LDBG(2) << "  Loop is invariant";
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
