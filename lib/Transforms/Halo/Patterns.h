#ifndef LIB_TRANSFORMS_HALO_PATTERNS_H_
#define LIB_TRANSFORMS_HALO_PATTERNS_H_

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project

namespace mlir {
namespace heir {

struct PeelPlaintextAffineForInit
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  PeelPlaintextAffineForInit(MLIRContext* context, DataFlowSolver* solver)
      : OpRewritePattern(context), solver(solver) {}

 public:
  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                PatternRewriter& rewriter) const override;

 private:
  DataFlowSolver* solver;
};

struct PeelPlaintextScfForInit : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  PeelPlaintextScfForInit(MLIRContext* context, DataFlowSolver* solver)
      : OpRewritePattern(context), solver(solver) {}

 public:
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter& rewriter) const override;

 private:
  DataFlowSolver* solver;
};

/// For docs, see the tablegen description in Passes.td for
/// bootstrap-loop-iter-args.
template <typename T>
struct BootstrapIterArgsPattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  BootstrapIterArgsPattern(MLIRContext* context, DataFlowSolver* solver)
      : OpRewritePattern<T>(context), solver(solver) {}

 public:
  LogicalResult matchAndRewrite(T forOp,
                                PatternRewriter& rewriter) const override {
    // Determine which inits are secret and need to be bootstrapped
    SmallVector<int> secretInitIndices;
    secretInitIndices.reserve(forOp.getInits().size());

    for (auto [i, init] : llvm::enumerate(forOp.getInits())) {
      if (isSecret(init, solver)) {
        secretInitIndices.push_back(i);
      }
    }

    if (secretInitIndices.empty()) {
      return rewriter.notifyMatchFailure(
          forOp, "No ciphertext values need bootstrapping");
    }

    // Insert a leading mgmt.level_reduce_min for each secret init
    rewriter.setInsertionPoint(forOp);
    for (auto i : secretInitIndices) {
      auto& initMutable = forOp.getInitsMutable()[i];
      auto reduceMinOp = mgmt::LevelReduceMinOp::create(
          rewriter, forOp.getLoc(), initMutable.get());
      rewriter.modifyOpInPlace(
          forOp, [&]() { initMutable.set(reduceMinOp.getResult()); });
    }

    // Insert a bootstrap for each secret yielded iter arg.
    rewriter.setInsertionPointToStart(forOp.getBody());
    for (auto i : secretInitIndices) {
      Value iterArg = forOp.getRegionIterArgs()[i];
      auto bootstrapOp =
          mgmt::BootstrapOp::create(rewriter, forOp.getLoc(), iterArg);
      rewriter.replaceAllUsesExcept(iterArg, bootstrapOp.getResult(),
                                    bootstrapOp);
    }

    // Insert a trailing mgmt.level_reduce_min for each secret yielded iter arg.
    Operation* yieldOp = forOp.getBody()->getTerminator();
    rewriter.setInsertionPoint(yieldOp);
    for (auto i : secretInitIndices) {
      auto& yieldedValue = yieldOp->getOpOperand(i);
      auto reduceMinOp = mgmt::LevelReduceMinOp::create(
          rewriter, forOp.getLoc(), yieldedValue.get());
      rewriter.modifyOpInPlace(
          yieldOp, [&]() { yieldedValue.set(reduceMinOp.getResult()); });
    }

    return success();
  }

 private:
  DataFlowSolver* solver;
};

struct PartialUnrollForLevelConsumptionAffineFor
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  PartialUnrollForLevelConsumptionAffineFor(MLIRContext* context,
                                            int forceMaxLevel,
                                            DataFlowSolver* solver)
      : OpRewritePattern(context),
        forceMaxLevel(forceMaxLevel),
        solver(solver) {}

 public:
  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                PatternRewriter& rewriter) const override;

  int forceMaxLevel;
  DataFlowSolver* solver;
};

struct PartialUnrollForLevelConsumptionSCFFor
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  PartialUnrollForLevelConsumptionSCFFor(MLIRContext* context,
                                         int forceMaxLevel,
                                         DataFlowSolver* solver)
      : OpRewritePattern(context),
        forceMaxLevel(forceMaxLevel),
        solver(solver) {}

 public:
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter& rewriter) const override;

  int forceMaxLevel;
  DataFlowSolver* solver;
};

// Remove any bootstrap ops that are marked for deletion in
// PartialUnrollForLevelConsumption.
struct DeleteAnnotatedOps : public RewritePattern {
  explicit DeleteAnnotatedOps(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

 public:
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_HALO_PATTERNS_H_
