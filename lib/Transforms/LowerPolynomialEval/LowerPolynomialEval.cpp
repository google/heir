#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h"

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "llvm/include/llvm/Support/Debug.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "lower-polynomial-eval"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERPOLYNOMIALEVAL
#include "lib/Transforms/LowerPolynomialEval/LowerPolynomialEval.h.inc"

struct LowerPolynomialEval
    : impl::LowerPolynomialEvalBase<LowerPolynomialEval> {
  using LowerPolynomialEvalBase::LowerPolynomialEvalBase;

  // FIXME: replace with actual lowering patterns
  struct DebugPrintPattern : public OpRewritePattern<polynomial::EvalOp> {
    DebugPrintPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<polynomial::EvalOp>(context) {}

    LogicalResult matchAndRewrite(polynomial::EvalOp op,
                                  PatternRewriter &rewriter) const override {
      LLVM_DEBUG(llvm::dbgs() << "Lowering polynomial::EvalOp: " << op << "\n");

      PolynomialEvalInterface evalInterface(op.getContext());
      Dialect *valueDialect = &op.getValue().getType().getDialect();
      if (!evalInterface.supportsPolynomial(op.getPolynomial(), valueDialect)) {
        return op.emitOpError()
               << "dialect for input type " << op.getValue().getType()
               << " does not support polynomial " << op.getPolynomial();
      }

      rewriter.modifyOpInPlace(
          op, [&]() { op->setAttr("lowered", rewriter.getUnitAttr()); });
      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DebugPrintPattern>(context);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
