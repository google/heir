#include <limits>
#include <utility>

#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_EXPANDCOPYPASS
#include "include/Conversion/MemrefToArith/MemrefToArith.h.inc"

namespace {

SmallVector<affine::AffineForOp> expandWithAffineLoops(OpBuilder& builder,
                                                       memref::CopyOp copy) {
  ImplicitLocOpBuilder b(copy.getLoc(), builder);

  // Create an affine for loop over the dimensions of the memref and
  // explicitly copy using affine loads and stores.
  MemRefType memRefType = cast<MemRefType>(copy.getSource().getType());
  SmallVector<mlir::Value, 4> indices;
  SmallVector<affine::AffineForOp> loops;

  auto zero = b.create<arith::ConstantIndexOp>(0);
  for (auto dim : memRefType.getShape()) {
    if (dim == 1) {
      // No need to create a loop for a one-dimensional index.
      indices.push_back(zero);
      continue;
    }
    auto loop = b.create<mlir::affine::AffineForOp>(0, dim);
    b.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
    loops.push_back(loop);
  }

  auto load = b.create<mlir::affine::AffineLoadOp>(copy.getSource(), indices);
  b.create<mlir::affine::AffineStoreOp>(load, copy.getTarget(), indices);
  return loops;
}

}  // namespace

// MemrefCopyExpansionPattern expands a `memref.copy` with explicit affine loads
// stores.
class MemrefCopyExpansionPattern
    : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
 public:
  MemrefCopyExpansionPattern(mlir::MLIRContext* context,
                             bool disableAffineLoops)
      : OpRewritePattern<memref::CopyOp>(context, /*benefit=*/3),
        disableAffineLoops_(disableAffineLoops) {}

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter& rewriter) const override {
    auto nestedLoops = expandWithAffineLoops(rewriter, copy);

    if (disableAffineLoops_ && !nestedLoops.empty()) {
      // nestedLoops[0].getBody(0)->walk<WalkOrder::PostOrder>(
      //     [&](affine::AffineForOp forOp) {
      //       auto unrollFactor =
      //           mlir::affine::getConstantTripCount(forOp).value_or(
      //               std::numeric_limits<int>::max());
      //       if (failed(loopUnrollUpToFactor(forOp, unrollFactor))) {
      //         return WalkResult::skip();
      //       }
      //       return WalkResult::advance();
      //     });
      //
      // Just unroll the outer loop for now.
      if (failed(loopUnrollFull(nestedLoops[0]))) {
        return mlir::failure();
      }
    }

    rewriter.eraseOp(copy);
    return mlir::success();
  }

 private:
  bool disableAffineLoops_;
};

// ExpandCopyPass intends to remove all memref copy operations.
struct ExpandCopyPass : impl::ExpandCopyPassBase<ExpandCopyPass> {
  using ExpandCopyPassBase::ExpandCopyPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MemrefCopyExpansionPattern>(&getContext(), disableAffineLoop);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
