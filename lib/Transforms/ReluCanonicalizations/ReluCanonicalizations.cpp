#include "lib/Transforms/ReluCanonicalizations/ReluCanonicalizations.h"

#include <utility>

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RELUCANONICALIZATIONS
#include "lib/Transforms/ReluCanonicalizations/ReluCanonicalizations.h.inc"

struct ReluCanonicalizations
    : impl::ReLUCanonicalizationsBase<ReluCanonicalizations> {
  using ReLUCanonicalizationsBase::ReLUCanonicalizationsBase;

  // Pattern for select(a >= c, a, c) = max(a, c)
  struct SelectOfCmpfPattern : public OpRewritePattern<arith::SelectOp> {
    using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::SelectOp op,
                                  PatternRewriter& rewriter) const override {
      // Condition should be a comparison
      auto cmpf = op.getCondition().getDefiningOp<arith::CmpFOp>();
      if (!cmpf)
        return rewriter.notifyMatchFailure(op, "condition is not a comparison");

      auto predicate = cmpf.getPredicate();
      if (predicate != arith::CmpFPredicate::UGT &&
          predicate != arith::CmpFPredicate::UGE)
        return rewriter.notifyMatchFailure(
            op, "comparison is not a > or >= operation");

      if (cmpf.getLhs() != op.getTrueValue())
        return rewriter.notifyMatchFailure(
            op, "lhs of comparison is not the select's true value");
      if (cmpf.getRhs() != op.getFalseValue())
        return rewriter.notifyMatchFailure(
            op, "rhs of comparison is not the select's false value");

      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, op.getTrueValue(),
                                                     op.getFalseValue());
      return success();
    }
  };

  // TODO: Implement other comparison patterns.
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SelectOfCmpfPattern>(context);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
