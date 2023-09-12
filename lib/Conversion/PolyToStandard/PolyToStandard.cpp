#include "include/Conversion/PolyToStandard/PolyToStandard.h"

#include "include/Dialect/Poly/IR/PolyOps.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "lib/Conversion/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "include/Conversion/PolyToStandard/PolyToStandard.h.inc"

class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](PolyType type) -> Type {
      RingAttr attr = type.getRing();
      uint32_t idealDegree = attr.ideal().getDegree();
      IntegerType elementTy =
          IntegerType::get(ctx, attr.coefficientModulus().getBitWidth(),
                           IntegerType::SignednessSemantics::Signless);
      // We must remove the ring attribute on the tensor, since the
      // unrealized_conversion_casts cannot carry the poly.ring attribute
      // through.
      return RankedTensorType::get({idealDegree}, elementTy);
    });

    // We don't include any custom materialization ops because this lowering is
    // all done in a single pass. The dialect conversion framework works by
    // resolving intermediate (mid-pass) type conflicts by inserting
    // unrealized_conversion_cast ops, and only converting those to custom
    // materializations if they persist at the end of the pass. In our case,
    // we'd only need to use custom materializations if we split this lowering
    // across multiple passes.
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultTy = typeConverter->convertType(op->getResultTypes()[0]);
    auto resultTensorTy = cast<RankedTensorType>(resultTy);
    auto resultShape = resultTensorTy.getShape()[0];
    auto resultEltTy = resultTensorTy.getElementType();

    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];
    auto inputEltTy = inputTensorTy.getElementType();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getOperands()[0];
    // Extend element type if needed.
    if (inputEltTy != resultEltTy) {
      // FromTensorOp verifies that the coefficient tensor's elements fit into
      // the polynomial.
      assert(inputEltTy.getIntOrFloatBitWidth() <
             resultEltTy.getIntOrFloatBitWidth());

      coeffValue = b.create<arith::ExtUIOp>(
          RankedTensorType::get(inputShape, resultEltTy), coeffValue);
    }

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
      coeffValue = b.create<tensor::PadOp>(
          resultTy, coeffValue, low, high,
          b.create<arith::ConstantOp>(rewriter.getIntegerAttr(resultEltTy, 0)),
          /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  ConvertToTensor(mlir::MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert add lowers a poly.add operation to arith operations. A poly.add
  // operation is defined within the polynomial ring. Coefficients are added
  // element-wise as elements of the ring, so they are performed modulo the
  // coefficient modulus.
  //
  // To perform modular addition, assume that `cmod` is the coefficient modulus
  // of the ring, and that `N` is the bitwidth used to store the ring elements.
  // This may be much larger than `log_2(cmod)`.
  //
  // Let `x` and `y` be the inputs to modular addition, then:
  //    c1, n1 = addui_extended(x, y)
  // If the coefficient modulus divides `2^N`, then return
  //    c0 = c1 % cmod
  // Otherwise, compute the adjusted result:
  //    c0 = ((c1 % cmod) + (n1 * 2^N % cmod)) % cmod
  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getLhs().getType();
    APInt mod = op.getType().cast<PolyType>().getRing().coefficientModulus();
    auto cmod = rewriter.create<arith::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(cast<ShapedType>(type), {mod}));

    auto addExtendedOp = rewriter.create<arith::AddUIExtendedOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    auto c1ModOp = rewriter.create<arith::RemUIOp>(
        op.getLoc(), addExtendedOp->getResult(0), cmod);
    // If mod divides 2^N, c1modOp is our result.
    if (mod.isPowerOf2()) {
      rewriter.replaceOp(op, c1ModOp.getResult());
      return success();
    }
    // Otherwise, add (n1 * 2^N % cmod)
    APInt quotient, remainder;
    APInt bigMod = APInt(mod.getBitWidth() + 1, 2) << (mod.getBitWidth() - 1);
    APInt::udivrem(bigMod, mod.zext(bigMod.getBitWidth()), quotient, remainder);
    remainder = remainder.trunc(mod.getBitWidth());

    auto bitwidth = rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        DenseElementsAttr::get(cast<ShapedType>(type), {remainder}));
    auto adjustOp =
        rewriter.create<arith::AddIOp>(op.getLoc(), c1ModOp, bitwidth);

    auto selectOp = rewriter.create<arith::SelectOp>(
        op.getLoc(), addExtendedOp.getResult(1), c1ModOp, adjustOp);
    // Mod the final result.
    rewriter.replaceOp(
        op, rewriter.create<arith::RemUIOp>(op.getLoc(), selectOp, cmod));

    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(mlir::MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(https://github.com/google/heir/issues/104): implement
    return success();
  }
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ConversionTarget target(*context);
    PolyToStandardTypeConverter typeConverter(context);

    target.addLegalDialect<arith::ArithDialect>();

    // target.addIllegalDialect<PolyDialect>();
    target.addIllegalOp<FromTensorOp, ToTensorOp, AddOp>();
    // target.addIllegalOp<AddOp>();
    // target.addIllegalOp<MulOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertFromTensor, ConvertToTensor, ConvertAdd>(typeConverter, context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    // TODO(https://github.com/google/heir/issues/143): Handle tensor of polys.
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace poly
}  // namespace heir
}  // namespace mlir
