#include "lib/Transforms/CompareToSignRewrite/CompareToSignRewrite.h"

#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_COMPARETOSIGNREWRITE
#include "lib/Transforms/CompareToSignRewrite/CompareToSignRewrite.h.inc"

// `elementType`, shaped like `like` if that is a shaped type.
static Type withShapeOf(Type like, Type elementType) {
  if (auto shaped = dyn_cast<ShapedType>(like))
    return shaped.clone(elementType);
  return elementType;
}

// A float constant of `type`, splat if `type` is shaped.
static Value floatConstant(PatternRewriter& rewriter, Location loc, Type type,
                           double value) {
  TypedAttr attr = rewriter.getFloatAttr(getElementTypeOrSelf(type), value);
  if (auto shaped = dyn_cast<ShapedType>(type))
    attr = DenseElementsAttr::get(shaped, attr);
  return arith::ConstantOp::create(rewriter, loc, attr);
}

struct CompareToSignRewrite
    : impl::CompareToSignRewriteBase<CompareToSignRewrite> {
  using CompareToSignRewriteBase::CompareToSignRewriteBase;

  // Pattern for arith.cmpf a < b and a > b,
  // assuming we can multiply with 1/2: a < b -> 0.5*(sign(b - a)+1)
  // Note that we ignore ordered/unordered comparison subtletlies here,
  // as this kind of arithmetization assumes no NaN/Inf values anyway.
  struct CmpFOpRewritePattern : public OpRewritePattern<arith::CmpFOp> {
    using OpRewritePattern<arith::CmpFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::CmpFOp op,
                                  PatternRewriter& rewriter) const override {
      Value lhs = op.getLhs();
      Value rhs = op.getRhs();
      Value sub;  // b - a for a < b, a - b for a > b

      if (op.getPredicate() == arith::CmpFPredicate::OLT ||
          op.getPredicate() == arith::CmpFPredicate::ULT)
        sub = arith::SubFOp::create(rewriter, op.getLoc(), rhs, lhs);
      else if (op.getPredicate() == arith::CmpFPredicate::OGT ||
               op.getPredicate() == arith::CmpFPredicate::UGT)
        sub = arith::SubFOp::create(rewriter, op.getLoc(), lhs, rhs);
      else
        // This pattern does not handle equality
        return failure();

      // Rewrite to 0.5 * (sign(<sub>) + 1)
      Value sign = math_ext::SignOp::create(rewriter, op.getLoc(), sub);
      Value one = floatConstant(rewriter, op.getLoc(), lhs.getType(), 1.0);
      Value half = floatConstant(rewriter, op.getLoc(), lhs.getType(), 0.5);
      Value result = arith::AddFOp::create(rewriter, op.getLoc(), sign, one);
      result = arith::MulFOp::create(rewriter, op.getLoc(), result, half);

      // Convert back to i1 for consistency with the original op
      result =
          arith::FPToSIOp::create(rewriter, op.getLoc(), op.getType(), result);

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
                                  PatternRewriter& rewriter) const override {
      Location loc = op.getLoc();
      Value lhs = op.getLhs();
      Value rhs = op.getRhs();

      if (op.getPredicate() != arith::CmpIPredicate::slt &&
          op.getPredicate() != arith::CmpIPredicate::ult &&
          op.getPredicate() != arith::CmpIPredicate::sgt &&
          op.getPredicate() != arith::CmpIPredicate::ugt)
        return failure();  // this pattern does not handle equality

      // (index types must be converted to i32 before we can cast to f32)
      Type i32Type = withShapeOf(lhs.getType(), rewriter.getI32Type());
      Type f32Type = withShapeOf(lhs.getType(), rewriter.getF32Type());
      if (getElementTypeOrSelf(lhs.getType()).isIndex())
        lhs = arith::IndexCastOp::create(rewriter, loc, i32Type, lhs);
      if (getElementTypeOrSelf(rhs.getType()).isIndex())
        rhs = arith::IndexCastOp::create(rewriter, loc, i32Type, rhs);
      //   Convert to f32 for the arithmetic
      lhs = arith::SIToFPOp::create(rewriter, loc, f32Type, lhs);
      rhs = arith::SIToFPOp::create(rewriter, loc, f32Type, rhs);

      Value sub;  // b - a for a < b, a - b for a > b
      if (op.getPredicate() == arith::CmpIPredicate::slt ||
          op.getPredicate() == arith::CmpIPredicate::ult)
        sub = arith::SubFOp::create(rewriter, loc, rhs, lhs);
      else if (op.getPredicate() == arith::CmpIPredicate::sgt ||
               op.getPredicate() == arith::CmpIPredicate::ugt)
        sub = arith::SubFOp::create(rewriter, loc, lhs, rhs);

      // Rewrite to 0.5 * (sign(<sub>) + 1)
      Value sign = math_ext::SignOp::create(rewriter, loc, sub);
      Value one = floatConstant(rewriter, loc, f32Type, 1.0);
      Value half = floatConstant(rewriter, loc, f32Type, 0.5);
      Value result = arith::AddFOp::create(rewriter, loc, sign, one);
      result = arith::MulFOp::create(rewriter, loc, result, half);

      // Convert back to i1 for consistency with the original op
      result = arith::FPToSIOp::create(rewriter, loc, op.getType(), result);

      rewriter.replaceOp(op, result);
      return success();
    }
  };

  // TODO (#1929): Implement patterns for other comparison operations

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<CmpFOpRewritePattern, CmpIOpRewritePattern>(context);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
