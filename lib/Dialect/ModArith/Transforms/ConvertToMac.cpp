#include "lib/Dialect/ModArith/Transforms/ConvertToMac.h"

#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DEF_CONVERTTOMAC
#include "lib/Dialect/ModArith/Transforms/Passes.h.inc"

struct FindMac : public OpRewritePattern<mod_arith::AddOp> {
  FindMac(mlir::MLIRContext* context)
      : OpRewritePattern<mod_arith::AddOp>(context) {}

  LogicalResult matchAndRewrite(mod_arith::AddOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Assume that we have a form a x b + rhs
    auto parent = op.getLhs().getDefiningOp<mod_arith::MulOp>();
    auto addOperand = op.getRhs();

    if (!parent) {
      auto parentRhs = op.getRhs().getDefiningOp<mod_arith::MulOp>();
      if (!parentRhs) {
        return failure();
      }
      // Find we have a form of lhs + a x b
      parent = parentRhs;
      addOperand = op.getLhs();
    }

    auto result =
        MacOp::create(b, parent.getLhs(), parent.getRhs(), addOperand);

    rewriter.replaceOp(op, result);

    if (parent.use_empty()) {
      rewriter.eraseOp(parent);
    }

    return success();
  }
};

struct ConvertToMac : impl::ConvertToMacBase<ConvertToMac> {
  using ConvertToMacBase::ConvertToMacBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FindMac>(context);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
