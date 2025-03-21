#include "lib/Transforms/ShapeInference/ShapeInference.h"

#include <memory>

#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "shape-inference"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SHAPEINFERENCE
#include "lib/Transforms/ShapeInference/ShapeInference.h.inc"

struct ConvertFuncArguments : public OpRewritePattern<func::FuncOp> {
  ConvertFuncArguments(MLIRContext *context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct ShapeInference : impl::ShapeInferenceBase<ShapeInference> {
  using ShapeInferenceBase::ShapeInferenceBase;

  // pattern to rewrite arguments of function of rankedtensor-with-dynamic-shape
  // type with an argument attribute "shape" that contains the desired shape
  // into fuinction arguments with a statically sized ranked tensor type.

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertFuncArguments>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
