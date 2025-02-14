#ifndef LIB_UTILS_REWRITEUTILS_REWRITEUTILS_H_
#define LIB_UTILS_REWRITEUTILS_REWRITEUTILS_H_

#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project

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

#endif  // LIB_UTILS_REWRITEUTILS_REWRITEUTILS_H_
