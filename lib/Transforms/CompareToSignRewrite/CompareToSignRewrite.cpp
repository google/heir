#include "lib/Transforms/CompareToSignRewrite/CompareToSignRewrite.h"

#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_COMPARETOSIGNREWRITE
#include "lib/Transforms/CompareToSignRewrite/CompareToSignRewrite.h.inc"

struct CompareToSignRewrite
    : impl::CompareToSignRewriteBase<CompareToSignRewrite> {
  using CompareToSignRewriteBase::CompareToSignRewriteBase;

  // Pattern for arith.cmpf a < b and b >a,
  // assuming we can multiply with 1/2: a < b -> 0.5*(sign(a - b)+1)
  // Note that we ignore ordered/unordered comparison subtletlies here,
  // as this kind of arithmetization assumes no NaN/Inf values anyway.
  struct CmpFOpRewritePattern : public OpRewritePattern<arith::CmpFOp> {
    using OpRewritePattern<arith::CmpFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::CmpFOp op,
                                  PatternRewriter &rewriter) const override {
      Value lhs = op.getLhs();
      Value rhs = op.getRhs();
      Value sub;  // will be set to a - b for a < b or b - a for b > a
      auto floatType = cast<FloatType>(lhs.getType());

      if (op.getPredicate() == arith::CmpFPredicate::OLT ||
          op.getPredicate() == arith::CmpFPredicate::ULT)
        sub = rewriter.create<arith::SubFOp>(op.getLoc(), lhs, rhs);
      else if (op.getPredicate() == arith::CmpFPredicate::OGT ||
               op.getPredicate() == arith::CmpFPredicate::UGT)
        sub = rewriter.create<arith::SubFOp>(op.getLoc(), rhs, lhs);
      else
        // This pattern does not handle equality
        return failure();

      // Rewrite to 0.5 * (sign(<sub>) + 1)
      Value sign = rewriter.create<math_ext::SignOp>(op.getLoc(), sub);
      auto oneAttr = rewriter.getFloatAttr(floatType, 1.0);
      Value one = rewriter.create<arith::ConstantOp>(op.getLoc(), oneAttr);
      auto halfAttr = rewriter.getFloatAttr(floatType, 0.5);
      Value half = rewriter.create<arith::ConstantOp>(op.getLoc(), halfAttr);
      Value result = rewriter.create<arith::AddFOp>(op.getLoc(), sign, one);
      result = rewriter.create<arith::MulFOp>(op.getLoc(), result, half);

      // Convert back to i1 for consistency with the original op
      result =
          rewriter.create<arith::FPToSIOp>(op.getLoc(), op.getType(), result);

      rewriter.replaceOp(op, result);
      return success();
    }
  };

  // Same as CmpFOpRewritePattern but for arith.cmpi
  // NOTE: This STILL assumes that we can multiply with 1/2,
  // and will convert the types to f32 before doing the arithmetic!
  // NOTE: This pattern treats unsigned and signed comparisons the same,
  // since the underlying arithmetic rewrite doesn't change anyway.
  // TODO (#1929): Find a way do differentiate between when we're using CKKS
  // and only have cmpi because of index/singless integer loop bounds,
  // and when we are in BGV/BFV world and can't use this pattern (no 0.5)
  struct CmpIOpRewritePattern : public OpRewritePattern<arith::CmpIOp> {
    using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::CmpIOp op,
                                  PatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      Value lhs = op.getLhs();
      Value rhs = op.getRhs();

      if (op.getPredicate() != arith::CmpIPredicate::slt &&
          op.getPredicate() != arith::CmpIPredicate::ult &&
          op.getPredicate() != arith::CmpIPredicate::sgt &&
          op.getPredicate() != arith::CmpIPredicate::ugt)
        return failure();  // this pattern does not handle equality

      // (index types must be converted to i32 before we can cast to f32)
      if (lhs.getType().isIndex())
        lhs = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                  lhs);
      if (rhs.getType().isIndex())
        rhs = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                  rhs);
      //   Convert to f32 for the arithmetic
      lhs = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF32Type(), lhs);
      rhs = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF32Type(), rhs);

      Value sub;  // will be set to a - b for a < b or b - a for b > a
      if (op.getPredicate() == arith::CmpIPredicate::slt ||
          op.getPredicate() == arith::CmpIPredicate::ult)
        sub = rewriter.create<arith::SubFOp>(loc, lhs, rhs);
      else if (op.getPredicate() == arith::CmpIPredicate::sgt ||
               op.getPredicate() == arith::CmpIPredicate::uge)
        sub = rewriter.create<arith::SubFOp>(loc, rhs, lhs);

      // Rewrite to 0.5 * (sign(<sub>) + 1)
      Value sign = rewriter.create<math_ext::SignOp>(loc, sub);
      Value one = rewriter.create<arith::ConstantFloatOp>(
          loc, rewriter.getF32Type(), APFloat(1.0f));

      Value half = rewriter.create<arith::ConstantFloatOp>(
          loc, rewriter.getF32Type(), APFloat(0.5f));
      Value result = rewriter.create<arith::AddFOp>(loc, sign, one);
      result = rewriter.create<arith::MulFOp>(loc, result, half);

      // Convert back to i1 for consistency with the original op
      result = rewriter.create<arith::FPToSIOp>(loc, op.getType(), result);

      rewriter.replaceOp(op, result);
      return success();
    }
  };

  // TODO (#1929): Implement patterns for other comparison operations

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<CmpFOpRewritePattern, CmpIOpRewritePattern>(context);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
