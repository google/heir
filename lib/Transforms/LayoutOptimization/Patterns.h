#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTPATTERNS_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTPATTERNS_H_

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

struct HoistArgLayouts : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter& rewriter) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTPATTERNS_H_
