#ifndef LIB_UTILS_REWRITEUTILS_REWRITEUTILS_H_
#define LIB_UTILS_REWRITEUTILS_REWRITEUTILS_H_

#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

// Simple Pattern that replaces an operation with a new operation
// assuming that no changes to types/operands/attributes are necessary
template <typename OriginalOp, typename NewOp>
struct Convert : public mlir::OpRewritePattern<OriginalOp> {
  Convert(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<OriginalOp>(context) {}

  llvm::LogicalResult matchAndRewrite(
      OriginalOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NewOp>(op, op->getOperands(), op->getAttrs());
    return llvm::success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arithmetic
/// operations using the StandardOps dialect.
class ExpandAffineApply : public OpRewritePattern<affine::AffineApplyOp> {
 public:
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                                llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap) return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_REWRITEUTILS_REWRITEUTILS_H_
