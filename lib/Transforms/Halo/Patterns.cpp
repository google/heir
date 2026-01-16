#include "lib/Transforms/Halo/Patterns.h"

#include <cmath>
#include <cstdint>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"  // from @llvm-project
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

}  // namespace heir
}  // namespace mlir
