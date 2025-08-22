#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTPATTERNS_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTPATTERNS_H_

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

// Try to fold the convert layout op into a preceding op if possible.
LogicalResult tryFoldLayoutConversionIntoPrevious(
    RewriterBase& rewriter, tensor_ext::ConvertLayoutOp op,
    SmallVector<Operation*>& opsToErase);

struct FoldLayoutConversions
    : public OpRewritePattern<tensor_ext::ConvertLayoutOp> {
  using OpRewritePattern<tensor_ext::ConvertLayoutOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(tensor_ext::ConvertLayoutOp op,
                                PatternRewriter& rewriter) const override;
};

struct HoistArgLayouts : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter& rewriter) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTPATTERNS_H_
