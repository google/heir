#include "lib/Conversion/ArithExtToArith/ArithExtToArith.h"

#include "lib/Dialect/ArithExt/IR/ArithExtOps.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith_ext {

#define GEN_PASS_DEF_ARITHEXTTOARITH
#include "lib/Conversion/ArithExtToArith/ArithExtToArith.h.inc"

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "lib/Conversion/ArithExtToArith/ArithExtToArith.cpp.inc"
}  // namespace rewrites

struct ConvertBarrettReduce : public OpConversionPattern<BarrettReduceOp> {
  ConvertBarrettReduce(mlir::MLIRContext *context)
      : OpConversionPattern<BarrettReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BarrettReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Compute B = 4^{bitWidth} and ratio = floordiv(B / modulus)
    auto input = adaptor.getInput();
    auto mod = APInt(64, op.getModulus());
    auto bitWidth = (mod - 1).getActiveBits();
    mod = mod.trunc(3 * bitWidth);
    auto B = APInt(3 * bitWidth, 1).shl(2 * bitWidth);
    auto barrettRatio = B.udiv(mod);

    Type intermediateType = IntegerType::get(b.getContext(), 3 * bitWidth);

    // Create our pre-computed constants
    TypedAttr ratioAttr, shiftAttr, modAttr;
    if (auto tensorType = dyn_cast<RankedTensorType>(input.getType())) {
      tensorType = tensorType.clone(tensorType.getShape(), intermediateType);
      ratioAttr = DenseElementsAttr::get(tensorType, barrettRatio);
      shiftAttr =
          DenseElementsAttr::get(tensorType, APInt(3 * bitWidth, 2 * bitWidth));
      modAttr = DenseElementsAttr::get(tensorType, mod);
      intermediateType = tensorType;
    } else if (auto integerType = dyn_cast<IntegerType>(input.getType())) {
      ratioAttr = IntegerAttr::get(intermediateType, barrettRatio);
      shiftAttr =
          IntegerAttr::get(intermediateType, APInt(3 * bitWidth, 2 * bitWidth));
      modAttr = IntegerAttr::get(intermediateType, mod);
    }

    auto ratioValue = b.create<arith::ConstantOp>(intermediateType, ratioAttr);
    auto shiftValue = b.create<arith::ConstantOp>(intermediateType, shiftAttr);
    auto modValue = b.create<arith::ConstantOp>(intermediateType, modAttr);

    // Intermediate value will be in the range [0,p^3) so we need to extend to
    // 3*bitWidth
    auto extendOp = b.create<arith::ExtUIOp>(intermediateType, input);

    // Compute x - floordiv(x * ratio, B) * mod
    auto mulRatioOp = b.create<arith::MulIOp>(extendOp, ratioValue);
    auto shrOp = b.create<arith::ShRUIOp>(mulRatioOp, shiftValue);
    auto mulModOp = b.create<arith::MulIOp>(shrOp, modValue);
    auto subOp = b.create<arith::SubIOp>(extendOp, mulModOp);

    auto truncOp = b.create<arith::TruncIOp>(input.getType(), subOp);

    rewriter.replaceOp(op, truncOp);

    return success();
  }
};

struct ArithExtToArith : impl::ArithExtToArithBase<ArithExtToArith> {
  using ArithExtToArithBase::ArithExtToArithBase;

  void runOnOperation() override;
};

void ArithExtToArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addIllegalDialect<ArithExtDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(context);
  patterns.add<ConvertBarrettReduce, rewrites::ConvertNormalised,
               rewrites::ConvertSubIfGE>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir
