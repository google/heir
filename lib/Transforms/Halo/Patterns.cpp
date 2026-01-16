#include "lib/Transforms/Halo/Patterns.h"

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h"  // @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"  // @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "halo-patterns"

namespace mlir {
namespace heir {

using affine::AffineForOp;

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

LogicalResult PartialUnrollForLevelConsumptionAffineFor::matchAndRewrite(
    affine::AffineForOp forOp, PatternRewriter& rewriter) const {
  SmallVector<Value> secretIterArgs;
  secretIterArgs.reserve(forOp.getInits().size());
  for (Value iterArg : forOp.getRegionIterArgs()) {
    if (isSecret(iterArg, solver)) {
      secretIterArgs.push_back(iterArg);
    }
  }

  if (secretIterArgs.empty()) {
    return rewriter.notifyMatchFailure(forOp, "No secret iter args detected");
  }

  // Now we need to compute how many loop iterations we can unroll.
  //
  // To start, assume we have just one iter arg. It is assumed to be the result
  // of a mgmt.level_reduce_min op just before being yielded. The quantity we
  // need to compute is the difference between the level just after the
  // bootstrap op and the level just before the level_reduce_min op.
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
  // Suppose this difference LEVEL_START - LEVEL_END is T. If T > LEVEL_END,
  // then we can unroll by a factor of floor(LEVEL_END / T).
  //
  // If we have many iter args, the allowable loop unroll factor is the minimum
  // among all iter args.
  //
  // FIXME: if there are two iter args, and the gap between their unroll factors
  // is very large, it may be worthwhile to insert extra bootstrap operations
  // for the "bad" iter arg so that you don't have to bootstrap both as often.
  // I think this will ultimately become a part of an improved bootstrapping
  // placement algorithm, where we can choose the loop unroll factor in part
  // based on what a bootstrapping placement algorithm would do when run on the
  // body of the loop.

  SmallVector<int> unrollFactors;
  for (Value iterArg : secretIterArgs) {
    ;
  }

  return success();
}

}  // namespace heir
}  // namespace mlir
