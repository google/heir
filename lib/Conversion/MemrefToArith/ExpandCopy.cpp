#include <memory>
#include <utility>

#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// MemrefCopyExpansionPass expands a `memref.copy` with explicit affine loads
// stores.
class MemrefCopyExpansionPass final
    : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto loc = copy.getLoc();
    auto memrefType = copy.getSource().getType().cast<MemRefType>();

    // Create an affine for loop over the dimensions of the memref and
    // explicitly copy using affine loads and stores.
    mlir::SmallVector<mlir::Value, 4> indices;
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto dim : memrefType.getShape()) {
      if (dim == 1) {
        // No need to create a loop for a one-dimensional index.
        indices.push_back(zero);
        continue;
      }
      auto loop = rewriter.create<mlir::affine::AffineForOp>(loc, 0, dim);
      rewriter.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
    }

    auto load = rewriter.create<mlir::affine::AffineLoadOp>(
        loc, copy.getSource(), indices);
    rewriter.create<mlir::affine::AffineStoreOp>(loc, load, copy.getTarget(),
                                                 indices);

    rewriter.eraseOp(copy);
    return mlir::success();
  }
};

// ExpandCopyPass intends to remove all memref copy operations.
struct ExpandCopyPass
    : public mlir::PassWrapper<ExpandCopyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<mlir::memref::CopyOp>();
    target.addLegalDialect<mlir::arith::ArithDialect,
                           mlir::affine::AffineDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MemrefCopyExpansionPass>(&getContext());

    (void)applyPartialConversion(getOperation(), target, std::move(patterns));
  }

  mlir::StringRef getArgument() const final { return "expand-copy"; }
};

std::unique_ptr<Pass> createExpandCopyPass() {
  return std::make_unique<ExpandCopyPass>();
}

}  // namespace heir
}  // namespace mlir
