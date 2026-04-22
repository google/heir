#include "lib/Dialect/Cheddar/Transforms/FuseOps.h"

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::cheddar {

//===----------------------------------------------------------------------===//
// Fusion patterns
//===----------------------------------------------------------------------===//

// Pattern: mult + relinearize + rescale -> hmult(rescale=true)
struct FuseMultRelinRescale : public OpRewritePattern<RescaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RescaleOp rescaleOp,
                                PatternRewriter &rewriter) const override {
    // Check that the input to rescale is a relinearize
    auto relinOp = rescaleOp.getInput().getDefiningOp<RelinearizeOp>();
    if (!relinOp || !relinOp.getResult().hasOneUse()) return failure();

    // Check that the input to relinearize is a mult
    auto multOp = relinOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp.getResult().hasOneUse()) return failure();

    // Fuse into HMult with rescale=true
    rewriter.replaceOpWithNewOp<HMultOp>(
        rescaleOp, rescaleOp.getOutput().getType(), multOp.getCtx(),
        multOp.getLhs(), multOp.getRhs(), relinOp.getMultKey(),
        /*rescale=*/rewriter.getBoolAttr(true));

    // Clean up now-dead ops
    rewriter.eraseOp(relinOp);
    rewriter.eraseOp(multOp);
    return success();
  }
};

// Pattern: mult + relinearize -> hmult(rescale=false)
struct FuseMultRelin : public OpRewritePattern<RelinearizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeOp relinOp,
                                PatternRewriter &rewriter) const override {
    // Don't match if this relin feeds into a rescale (handled by
    // FuseMultRelinRescale)
    if (relinOp.getResult().hasOneUse()) {
      auto *user = *relinOp.getResult().getUsers().begin();
      if (isa<RescaleOp>(user)) return failure();
    }

    auto multOp = relinOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp.getResult().hasOneUse()) return failure();

    rewriter.replaceOpWithNewOp<HMultOp>(
        relinOp, relinOp.getOutput().getType(), multOp.getCtx(),
        multOp.getLhs(), multOp.getRhs(), relinOp.getMultKey(),
        /*rescale=*/rewriter.getBoolAttr(false));

    rewriter.eraseOp(multOp);
    return success();
  }
};

// Pattern: mult + relinearize_rescale -> hmult(rescale=true)
// (In case we already have a fused relin+rescale but not the full triple)
struct FuseMultRelinRescaleFused
    : public OpRewritePattern<RelinearizeRescaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeRescaleOp relinRescaleOp,
                                PatternRewriter &rewriter) const override {
    auto multOp = relinRescaleOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp.getResult().hasOneUse()) return failure();

    rewriter.replaceOpWithNewOp<HMultOp>(
        relinRescaleOp, relinRescaleOp.getOutput().getType(), multOp.getCtx(),
        multOp.getLhs(), multOp.getRhs(), relinRescaleOp.getMultKey(),
        /*rescale=*/rewriter.getBoolAttr(true));

    rewriter.eraseOp(multOp);
    return success();
  }
};

// Pattern: hrot(a) + b -> hrot_add(a, b)
// Matches: %rotated = cheddar.hrot ...; %sum = cheddar.add %ctx, %rotated, %b
struct FuseHRotAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is an hrot with a single use
    HRotOp hrotOp = nullptr;
    Value otherOperand;

    if (auto lhsHrot = addOp.getLhs().getDefiningOp<HRotOp>()) {
      if (lhsHrot.getResult().hasOneUse()) {
        hrotOp = lhsHrot;
        otherOperand = addOp.getRhs();
      }
    }
    if (!hrotOp) {
      if (auto rhsHrot = addOp.getRhs().getDefiningOp<HRotOp>()) {
        if (rhsHrot.getResult().hasOneUse()) {
          hrotOp = rhsHrot;
          otherOperand = addOp.getLhs();
        }
      }
    }
    if (!hrotOp) return failure();

    // Only fuse static-shift rotations into HRotAdd
    auto staticShift = hrotOp.getStaticShift();
    if (!staticShift) return failure();

    rewriter.replaceOpWithNewOp<HRotAddOp>(
        addOp, addOp.getOutput().getType(), hrotOp.getCtx(), hrotOp.getInput(),
        otherOperand, hrotOp.getRotKey(), *staticShift);

    rewriter.eraseOp(hrotOp);
    return success();
  }
};

// Pattern: hconj(a) + b -> hconj_add(a, b)
struct FuseHConjAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    HConjOp hconjOp = nullptr;
    Value otherOperand;

    if (auto lhsHconj = addOp.getLhs().getDefiningOp<HConjOp>()) {
      if (lhsHconj.getResult().hasOneUse()) {
        hconjOp = lhsHconj;
        otherOperand = addOp.getRhs();
      }
    }
    if (!hconjOp) {
      if (auto rhsHconj = addOp.getRhs().getDefiningOp<HConjOp>()) {
        if (rhsHconj.getResult().hasOneUse()) {
          hconjOp = rhsHconj;
          otherOperand = addOp.getLhs();
        }
      }
    }
    if (!hconjOp) return failure();

    rewriter.replaceOpWithNewOp<HConjAddOp>(
        addOp, addOp.getOutput().getType(), hconjOp.getCtx(),
        hconjOp.getInput(), otherOperand, hconjOp.getConjKey());

    rewriter.eraseOp(hconjOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_CHEDDARFUSEOPS
#include "lib/Dialect/Cheddar/Transforms/FuseOps.h.inc"

struct CheddarFuseOps : public impl::CheddarFuseOpsBase<CheddarFuseOps> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Fusion patterns. Order matters: try the longest fusions first.
    patterns.add<FuseMultRelinRescale>(context, /*benefit=*/3);
    patterns.add<FuseMultRelinRescaleFused>(context, /*benefit=*/2);
    patterns.add<FuseMultRelin>(context, /*benefit=*/1);
    patterns.add<FuseHRotAdd>(context, /*benefit=*/1);
    patterns.add<FuseHConjAdd>(context, /*benefit=*/1);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::cheddar
