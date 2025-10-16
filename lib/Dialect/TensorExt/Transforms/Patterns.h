#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_PATTERNS_H_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_PATTERNS_H_

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

struct FoldConvertLayoutIntoAssignLayoutPattern
    : public OpRewritePattern<AssignLayoutOp> {
  using OpRewritePattern<AssignLayoutOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(AssignLayoutOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_PATTERNS_H_
