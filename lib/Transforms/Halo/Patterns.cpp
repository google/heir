#include "lib/Transforms/Halo/Patterns.h"

#include <cmath>
#include <cstdint>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "halo-patterns"

namespace mlir {
namespace heir {

using affine::AffineForOp;
using affine::loopUnrollByFactor;

// Test if the two lists of values have mismatching secretness. Nb., a secret
// init with a non-secret iter arg is probably not a meaningful outcome of this
// function, but it doesn't hurt to support it.
static bool hasMismatch(ArrayRef<Value> inits, ArrayRef<Value> yieldedValues,
                        DataFlowSolver* solver) {
  for (int i = 0; i < inits.size(); ++i) {
    Value init = inits[i];
    Value yield = yieldedValues[i];

    auto* initLattice = solver->lookupState<SecretnessLattice>(init);
    if (!initLattice || !initLattice->getValue().isInitialized()) {
      continue;
    }

    auto* yieldLattice = solver->lookupState<SecretnessLattice>(yield);
    if (!yieldLattice || !yieldLattice->getValue().isInitialized()) {
      continue;
    }

    if (initLattice->getValue().getSecretness() !=
        yieldLattice->getValue().getSecretness()) {
      return true;
    }
  }

  return false;
}

LogicalResult PeelPlaintextAffineForInit::matchAndRewrite(
    AffineForOp forOp, PatternRewriter& rewriter) const {
  // Determine if the inits are public while the yielded values are secret
  SmallVector<Value> inits(forOp.getInits());
  SmallVector<Value> yieldedValues(forOp.getYieldedValues());
  bool mismatch = hasMismatch(inits, yieldedValues, solver);

  if (!mismatch) {
    return rewriter.notifyMatchFailure(
        forOp, "No ciphertext/plaintext mismatch detected");
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Found mismatch between init types and iter arg types\n");

  if (!forOp.hasConstantBounds())
    return rewriter.notifyMatchFailure(
        forOp, "Only affine.for ops with constant bounds are supported");

  int64_t lb = forOp.getConstantLowerBound();
  int64_t ub = forOp.getConstantUpperBound();
  int64_t step = forOp.getStep().getSExtValue();
  int64_t splitBound = lb + step;

  if (ceil(float(ub - lb) / step) <= 1) {
    return rewriter.notifyMatchFailure(
        forOp, "Peeling is not needed if there is one or less iteration.");
  }

  // Split the loop into two loops, the first of which has just one iteration,
  // and then unroll the single-iteration loop.
  AffineForOp firstIteration =
      cast<AffineForOp>(rewriter.clone(*forOp.getOperation()));
  rewriter.modifyOpInPlace(firstIteration, [&]() {
    firstIteration.setConstantUpperBound(splitBound);
  });

  rewriter.modifyOpInPlace(forOp, [&]() {
    forOp.getInitsMutable().assign(firstIteration->getResults());
    forOp.setConstantLowerBound(splitBound);
  });

  rewriter.modifyOpInPlace(firstIteration, [&]() {
    if (failed(loopUnrollFull(firstIteration))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to unroll single-iteration affine.for!\n");
    }
  });
  return success();
}

LogicalResult PeelPlaintextScfForInit::matchAndRewrite(
    scf::ForOp forOp, PatternRewriter& rewriter) const {
  // Determine if the inits are public while the yielded values are secret
  SmallVector<Value> inits(forOp.getInits());
  SmallVector<Value> yieldedValues(forOp.getYieldedValues());
  bool mismatch = hasMismatch(inits, yieldedValues, solver);

  if (!mismatch) {
    return rewriter.notifyMatchFailure(
        forOp, "No ciphertext/plaintext mismatch detected");
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Found mismatch between init types and iter arg types\n");

  scf::ForOp firstIteration;
  LogicalResult result =
      peelForLoopFirstIteration(rewriter, forOp, firstIteration);
  if (failed(result)) {
    return failure();
  }

  rewriter.modifyOpInPlace(firstIteration, [&]() {
    if (failed(loopUnrollFull(firstIteration))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to unroll single-iteration affine.for!\n");
    }
  });

  return success();
}

// Helper for rejecting a loop unroll pattern. Check:
//
// - all secret iter args are immediately bootstrapped and have no other uses
// - all secret iter args level_reduce_min just before being yielded
//
// If successful, the SSA values corresponding to secret iter args are returned.
template <typename ForLoop>
FailureOr<SmallVector<Value>> isLoopStructuredForHaloUnroll(
    ForLoop forOp, DataFlowSolver* solver) {
  SmallVector<Value> secretIterArgs;

  for (Value iterArg : forOp.getRegionIterArgs()) {
    if (isSecret(iterArg, solver)) {
      if (!iterArg.hasOneUse()) {
        return failure();
      }
      if (!isa<mgmt::BootstrapOp>(*iterArg.getUsers().begin())) {
        return failure();
      }

      Value yieldedValue =
          forOp.getTiedLoopYieldedValue(cast<BlockArgument>(iterArg))->get();
      if (!isa<mgmt::LevelReduceMinOp>(yieldedValue.getDefiningOp())) {
        return failure();
      }

      secretIterArgs.push_back(iterArg);
    }
  }

  if (secretIterArgs.empty()) {
    return failure();
  }

  return secretIterArgs;
}

// Inject an scf::ForOp overload of this function, which exists upstream for
// affine already.
FailureOr<int64_t> getConstantTripCount(scf::ForOp forOp) {
  if (auto step = forOp.getConstantStep();
      !step.has_value() || !step->isOne()) {
    if (step.has_value()) {
      return mlir::failure();
    }
    return mlir::failure();
  }
  APInt lowerBound;
  if (!matchPattern(forOp.getLowerBound(), m_ConstantInt(&lowerBound))) {
    return mlir::failure();
  }

  APInt upperBound;
  if (!matchPattern(forOp.getUpperBound(), m_ConstantInt(&upperBound))) {
    return mlir::failure();
  }

  return (upperBound - lowerBound).getLimitedValue();
}

template <typename ForOp>
LogicalResult doPartialUnroll(ForOp forOp, PatternRewriter& rewriter,
                              int forceMaxLevel, DataFlowSolver* solver) {
  FailureOr<SmallVector<Value>> secretIterArgsResult =
      isLoopStructuredForHaloUnroll(forOp, solver);
  if (failed(secretIterArgsResult)) {
    return rewriter.notifyMatchFailure(forOp, "Loop preconditions not met");
  }
  SmallVector<Value> secretIterArgs = secretIterArgsResult.value();

  std::optional<uint64_t> maybeTripCount = getConstantTripCount(forOp);
  if (!maybeTripCount.has_value()) return failure();
  int64_t tripCount = maybeTripCount.value();

  // Now we need to compute how many loop iterations we can unroll.
  //
  // To start, assume we have just one iter arg. It is assumed to be the result
  // of a mgmt.level_reduce_min op just before being yielded. The quantity we
  // need to compute is the difference between the level just after the
  // bootstrap op (the "effective level" of bootstrap) and the level just before
  // the level_reduce_min op.
  //
  //  %2 = mgmt.level_reduce_min %1 : i32
  //  %3 = affine.for %arg1 = 1 to 12 iter_args(%arg2 = %2) -> (i32) {
  //     %4 = mgmt.bootstrap %arg2 : i32
  //     LEVEL_START[%arg2] = level(%4)
  //
  //     ...
  //
  //     LEVEL_END[%arg2] = level(%5)
  //     %6 = mgmt.level_reduce_min %5 : i32
  //     affine.yield %6 : i32
  //  }
  //
  // If LEVEL_END > 0, then we can unroll if and only if LEVEL_START / T > 1,
  // and (int) LEVEL_START / T is the factor we can unroll by.
  //
  // If we have many iter args, the allowable loop unroll factor is the minimum
  // among all iter args.
  //
  // TODO(#2556): Use smarter unroll strategies for multiple iter_args.

  SmallVector<int> unrollFactors;
  for (Value iterArg : secretIterArgs) {
    auto bootstrapOp = cast<mgmt::BootstrapOp>(*iterArg.getUsers().begin());
    auto* levelStart =
        solver->lookupState<LevelLattice>(bootstrapOp.getResult());

    Value yieldedValue =
        forOp.getTiedLoopYieldedValue(cast<BlockArgument>(iterArg))->get();
    mgmt::LevelReduceMinOp levelReduceOp =
        cast<mgmt::LevelReduceMinOp>(yieldedValue.getDefiningOp());
    auto* levelEnd =
        solver->lookupState<LevelLattice>(levelReduceOp.getInput());

    if (!levelStart || !levelEnd || !levelStart->getValue().isInt() ||
        !levelEnd->getValue().isInt()) {
      return rewriter.notifyMatchFailure(
          forOp,
          "Start and end levels were not inferable to be concrete integers");
    }

    // In the LevelAnalysis, the levels start from 0 and go up, so the
    // difference is the ending level minus the starting level.
    int levelEndVal = levelEnd->getValue().getInt();
    int levelStartVal = levelStart->getValue().getInt();

    // TODO(#2557): consider effective bootstrap level
    int levelAfterBootstrap =
        getMaxLevel(forOp->template getParentOfType<func::FuncOp>(), solver);
    if (forceMaxLevel > 0) {
      levelAfterBootstrap = forceMaxLevel;
      LLVM_DEBUG(llvm::dbgs()
                 << "Using forced max level of " << forceMaxLevel << "\n");
    }
    int levelsUsedInLoop = levelEndVal - levelStartVal;
    if (levelAfterBootstrap / levelsUsedInLoop > 1) {
      unrollFactors.push_back(levelAfterBootstrap / levelsUsedInLoop);
    }
  }

  if (unrollFactors.empty()) {
    return rewriter.notifyMatchFailure(forOp,
                                       "No unroll factors could be found.");
  }

  int chosenUnrollFactor = *llvm::min_element(unrollFactors);
  chosenUnrollFactor =
      tripCount < chosenUnrollFactor ? tripCount : chosenUnrollFactor;
  if (chosenUnrollFactor > 1) {
    // The function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn that
    // we pass to the loop unroll step ensures that we can tell which bootstrap
    // ops and level_reduce_min ops are safe to remove post-unroll.
    LLVM_DEBUG(llvm::dbgs() << "Applying Halo-style unroll by a factor of "
                            << chosenUnrollFactor << "\n");

    // First mark the special loop invariance ops with an attribute, so that any
    // other operations of the same type in the loop body are not accidentally
    // deleted by cleanup. The name of the attribute is arbitrary as it will be
    // cleaned up before this pattern is complete.
    std::string specialOpKey = "halo.invariance";
    for (Value iterArg : forOp.getRegionIterArgs()) {
      if (isSecret(iterArg, solver)) {
        auto bootstrapOp = cast<mgmt::BootstrapOp>(*iterArg.getUsers().begin());
        rewriter.modifyOpInPlace(bootstrapOp, [&]() {
          bootstrapOp->setAttr(specialOpKey, rewriter.getUnitAttr());
        });

        Value yieldedValue =
            forOp.getTiedLoopYieldedValue(cast<BlockArgument>(iterArg))->get();
        auto levelReduceOp =
            cast<mgmt::LevelReduceMinOp>(yieldedValue.getDefiningOp());
        rewriter.modifyOpInPlace(levelReduceOp, [&]() {
          levelReduceOp->setAttr(specialOpKey, rewriter.getUnitAttr());
        });
      }
    }

    if (failed(loopUnrollByFactor(
            forOp, chosenUnrollFactor,
            [&](unsigned index, Operation* clonedOp, OpBuilder builder) {
              if (!clonedOp->hasAttr(specialOpKey)) return;

              // Internally, loopUnrollByFactor starts the passed `index` at 1
              // and goes up to unrollFactor (exclusive). This index 1
              // corresponds to the second loop iteration, but this callable is
              // never called for index 0 because the original loop body
              // is left intact and the additional unrolled iterations are
              // inserted after it. However, this is an implementation detail
              // that we try not to rely on.
              if (index > 0 && isa<mgmt::BootstrapOp>(clonedOp)) {
                // We cannot remove the op in the middle of the loop unrolling
                // process, so instead mark it for later removal.
                rewriter.modifyOpInPlace(clonedOp, [&]() {
                  clonedOp->setAttr("halo.remove", builder.getUnitAttr());
                });
              } else if (index < chosenUnrollFactor - 1 &&
                         isa<mgmt::LevelReduceMinOp>(clonedOp)) {
                rewriter.modifyOpInPlace(clonedOp, [&]() {
                  clonedOp->setAttr("halo.remove", builder.getUnitAttr());
                });
              }
              rewriter.modifyOpInPlace(
                  clonedOp, [&]() { clonedOp->removeAttr(specialOpKey); });
            })))
      return failure();
  }

  return success();
}

LogicalResult PartialUnrollForLevelConsumptionAffineFor::matchAndRewrite(
    affine::AffineForOp forOp, PatternRewriter& rewriter) const {
  return doPartialUnroll(forOp, rewriter, forceMaxLevel, solver);
}

LogicalResult PartialUnrollForLevelConsumptionSCFFor::matchAndRewrite(
    scf::ForOp forOp, PatternRewriter& rewriter) const {
  return doPartialUnroll(forOp, rewriter, forceMaxLevel, solver);
}

LogicalResult DeleteAnnotatedOps::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  if (!op->hasAttrOfType<UnitAttr>("halo.remove")) {
    return failure();
  }

  if (op->getNumResults() != 1 || op->getNumOperands() != 1) {
    return failure();
  }

  rewriter.replaceOp(op, op->getOperand(0));
  return success();
}

}  // namespace heir
}  // namespace mlir
