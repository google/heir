#include "lib/Dialect/Mgmt/IR/MgmtOps.h"

#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtPatterns.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

struct ModReduceAfterLevelReduce : public OpRewritePattern<LevelReduceOp> {
  using OpRewritePattern<LevelReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LevelReduceOp op,
                                PatternRewriter& rewriter) const override {
    auto modReduceOp = op.getInput().getDefiningOp<ModReduceOp>();
    if (!modReduceOp || !modReduceOp->hasOneUse() || !op->hasOneUse())
      return failure();

    Value input = modReduceOp.getInput();
    int64_t levelToDrop = op.getLevelToDrop();

    auto oldLrAttr = findMgmtAttrAssociatedWith(op.getResult());
    auto oldMrAttr = findMgmtAttrAssociatedWith(modReduceOp.getResult());
    auto inputAttr = findMgmtAttrAssociatedWith(input);

    auto newLevelReduceOp =
        rewriter.create<LevelReduceOp>(op.getLoc(), input, levelToDrop);

    auto newModReduceOp =
        rewriter.create<ModReduceOp>(op.getLoc(), newLevelReduceOp.getResult());

    if (oldLrAttr) {
      setMgmtAttrAssociatedWith(newModReduceOp.getResult(), oldLrAttr);
    }

    if (inputAttr && oldMrAttr) {
      int64_t newLrLevel = 0;
      if (oldLrAttr) {
        newLrLevel = oldLrAttr.getLevel() + 1;
      } else {
        newLrLevel = inputAttr.getLevel() - levelToDrop;
      }

      auto newLrAttr =
          MgmtAttr::get(op.getContext(), newLrLevel, inputAttr.getDimension(),
                        inputAttr.getScale());
      setMgmtAttrAssociatedWith(newLevelReduceOp.getResult(), newLrAttr);
    }

    rewriter.replaceOp(op, newModReduceOp.getResult());
    return success();
  }
};

struct ModReduceAfterAdjustScale : public OpRewritePattern<AdjustScaleOp> {
  using OpRewritePattern<AdjustScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AdjustScaleOp op,
                                PatternRewriter& rewriter) const override {
    auto modReduceOp = op.getInput().getDefiningOp<ModReduceOp>();
    if (!modReduceOp || !modReduceOp->hasOneUse() || !op->hasOneUse())
      return failure();

    Value input = modReduceOp.getInput();
    auto id = op.getIdAttr();

    auto oldAsAttr = findMgmtAttrAssociatedWith(op.getResult());
    auto oldMrAttr = findMgmtAttrAssociatedWith(modReduceOp.getResult());
    auto inputAttr = findMgmtAttrAssociatedWith(input);

    auto newAdjustScaleOp =
        rewriter.create<AdjustScaleOp>(op.getLoc(), input, id);

    auto newModReduceOp =
        rewriter.create<ModReduceOp>(op.getLoc(), newAdjustScaleOp.getResult());

    if (oldAsAttr) {
      setMgmtAttrAssociatedWith(newModReduceOp.getResult(), oldAsAttr);
    }

    if (inputAttr && oldMrAttr && oldAsAttr) {
      int64_t s_in = inputAttr.getScale();
      int64_t s_mr = oldMrAttr.getScale();
      int64_t s_as = oldAsAttr.getScale();
      if (s_mr == 0) return failure();
      int64_t newScale = s_as * s_in / s_mr;

      auto newAsAttr = MgmtAttr::get(op.getContext(), inputAttr.getLevel(),
                                     inputAttr.getDimension(), newScale);
      setMgmtAttrAssociatedWith(newAdjustScaleOp.getResult(), newAsAttr);
    }

    rewriter.replaceOp(op, newModReduceOp.getResult());
    return success();
  }
};

struct AdjustScaleAfterLevelReduce : public OpRewritePattern<LevelReduceOp> {
  using OpRewritePattern<LevelReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LevelReduceOp op,
                                PatternRewriter& rewriter) const override {
    auto adjustScaleOp = op.getInput().getDefiningOp<AdjustScaleOp>();
    if (!adjustScaleOp || !adjustScaleOp->hasOneUse() || !op->hasOneUse())
      return failure();

    Value input = adjustScaleOp.getInput();
    auto levelToDrop = op.getLevelToDrop();
    auto id = adjustScaleOp.getIdAttr();

    auto oldLrAttr = findMgmtAttrAssociatedWith(op.getResult());
    auto inputAttr = findMgmtAttrAssociatedWith(input);

    auto newLevelReduceOp =
        rewriter.create<LevelReduceOp>(op.getLoc(), input, levelToDrop);

    auto newAdjustScaleOp = rewriter.create<AdjustScaleOp>(
        op.getLoc(), newLevelReduceOp.getResult(), id);

    if (oldLrAttr) {
      setMgmtAttrAssociatedWith(newAdjustScaleOp.getResult(), oldLrAttr);
    }

    if (inputAttr && oldLrAttr) {
      auto newLrAttr =
          MgmtAttr::get(op.getContext(), oldLrAttr.getLevel(),
                        inputAttr.getDimension(), inputAttr.getScale());
      setMgmtAttrAssociatedWith(newLevelReduceOp.getResult(), newLrAttr);
    }

    rewriter.replaceOp(op, newAdjustScaleOp.getResult());
    return success();
  }
};

struct MergeLevelReduce : public OpRewritePattern<LevelReduceOp> {
  using OpRewritePattern<LevelReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LevelReduceOp op,
                                PatternRewriter& rewriter) const override {
    auto innerLr = op.getInput().getDefiningOp<LevelReduceOp>();
    if (!innerLr || !innerLr->hasOneUse()) return failure();

    Value input = innerLr.getInput();
    int64_t levelToDrop = op.getLevelToDrop() + innerLr.getLevelToDrop();

    auto oldLr2Attr = findMgmtAttrAssociatedWith(op.getResult());

    auto newLr =
        rewriter.create<LevelReduceOp>(op.getLoc(), input, levelToDrop);

    if (oldLr2Attr) {
      setMgmtAttrAssociatedWith(newLr.getResult(), oldLr2Attr);
    }

    rewriter.replaceOp(op, newLr.getResult());
    return success();
  }
};

struct MergeModReduce : public OpRewritePattern<ModReduceOp> {
  using OpRewritePattern<ModReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModReduceOp op,
                                PatternRewriter& rewriter) const override {
    auto innerMr = op.getInput().getDefiningOp<ModReduceOp>();
    if (!innerMr || !innerMr->hasOneUse()) return failure();

    Value input = innerMr.getInput();

    auto oldMr2Attr = findMgmtAttrAssociatedWith(op.getResult());
    auto inputAttr = findMgmtAttrAssociatedWith(input);

    auto newLevelReduceOp =
        rewriter.create<LevelReduceOp>(op.getLoc(), input, /*levelToDrop*/ 1);

    auto newModReduceOp =
        rewriter.create<ModReduceOp>(op.getLoc(), newLevelReduceOp.getResult());

    if (oldMr2Attr) {
      setMgmtAttrAssociatedWith(newModReduceOp.getResult(), oldMr2Attr);
    }

    if (inputAttr) {
      auto newLrAttr =
          MgmtAttr::get(op.getContext(), inputAttr.getLevel() - 1,
                        inputAttr.getDimension(), inputAttr.getScale());
      setMgmtAttrAssociatedWith(newLevelReduceOp.getResult(), newLrAttr);
    }

    rewriter.replaceOp(op, newModReduceOp.getResult());
    return success();
  }
};

struct MergeAdjustScale : public OpRewritePattern<AdjustScaleOp> {
  using OpRewritePattern<AdjustScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AdjustScaleOp op,
                                PatternRewriter& rewriter) const override {
    auto innerAs = op.getInput().getDefiningOp<AdjustScaleOp>();
    if (!innerAs || !innerAs->hasOneUse()) return failure();

    Value input = innerAs.getInput();
    auto id2 = op.getIdAttr();

    auto oldAs2Attr = findMgmtAttrAssociatedWith(op.getResult());

    auto newAs = rewriter.create<AdjustScaleOp>(op.getLoc(), input, id2);

    if (oldAs2Attr) {
      setMgmtAttrAssociatedWith(newAs.getResult(), oldAs2Attr);
    }

    rewriter.replaceOp(op, newAs.getResult());
    return success();
  }
};

void ModReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<MergeModReduce>(context);
}

void LevelReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<MergeLevelReduce>(context);
  results.add<ModReduceAfterLevelReduce>(context);
  results.add<AdjustScaleAfterLevelReduce>(context);
}

void AdjustScaleOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<ModReduceAfterAdjustScale, MergeAdjustScale>(context);
}

void LevelReduceMinOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                   MLIRContext* context) {
  results.add<ReplaceWithLevelReduce>(context);
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

void cleanupInitOp(Operation* top) {
  top->walk([&](mgmt::InitOp initOp) {
    initOp.getOutput().replaceAllUsesWith(initOp.getInput());
    initOp.erase();
  });
}

int LevelReduceOp::getLevelsToDrop() { return getLevelToDrop(); }

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
