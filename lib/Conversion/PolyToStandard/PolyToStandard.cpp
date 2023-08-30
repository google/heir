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
  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getLhs().getType();
    auto addOp = rewriter.create<arith::AddIOp>(
        op.getLoc(), type, adaptor.getLhs(), adaptor.getRhs());

    APInt mod = op.getType().getRing().coefficientModulus();
    assert(mod != 0 && "coefficient modulus must not be zero");
    auto modConstOp = rewriter.create<arith::ConstantOp>(
        addOp.getLoc(), DenseElementsAttr::get(type.cast<ShapedType>(), {mod}));

    auto modOp = rewriter.create<arith::RemUIOp>(
        modConstOp.getLoc(), addOp.getResult(), modConstOp.getResult());
    rewriter.replaceOp(op, modOp.getResult());

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
    target.addIllegalOp<FromTensorOp, ToTensorOp>();
    // target.addIllegalOp<AddOp>();
    // target.addIllegalOp<MulOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertFromTensor, ConvertToTensor>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);
    target.addIllegalOp<PolyFromCoeffsOp, AddOp>();
    // target.addIllegalOp<MulOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertPolyFromCoeffs, ConvertAdd>(typeConverter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace poly
}  // namespace heir
}  // namespace mlir
