#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::arith {

#define GEN_PASS_DEF_ARITHTOCGGI
#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h.inc"

static lwe::LWECiphertextType convertArithToCGGIType(IntegerType type,
                                                     MLIRContext *ctx) {
  return lwe::LWECiphertextType::get(ctx,
                                     lwe::UnspecifiedBitFieldEncodingAttr::get(
                                         ctx, type.getIntOrFloatBitWidth()),
                                     lwe::LWEParamsAttr());
  ;
}

static Type convertArithLikeToCGGIType(ShapedType type, MLIRContext *ctx) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    return type.cloneWith(type.getShape(),
                          convertArithToCGGIType(arithType, ctx));
  }
  return type;
}

class ArithToCGGITypeConverter : public TypeConverter {
 public:
  ArithToCGGITypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // Convert Integer types to LWE ciphertext types
    addConversion([ctx](IntegerType type) -> Type {
      return convertArithToCGGIType(type, ctx);
    });

    addConversion([ctx](ShapedType type) -> Type {
      return convertArithLikeToCGGIType(type, ctx);
    });
  }
};

struct ConvertConstantOp : public OpConversionPattern<mlir::arith::ConstantOp> {
  ConvertConstantOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isa<IndexType>(op.getValue().getType())) {
      return failure();
    }
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto intValue = cast<IntegerAttr>(op.getValue()).getValue().getSExtValue();
    auto inputValue = mlir::IntegerAttr::get(op.getType(), intValue);

    auto encoding = lwe::UnspecifiedBitFieldEncodingAttr::get(
        op->getContext(), op.getValue().getType().getIntOrFloatBitWidth());
    auto lweType = lwe::LWECiphertextType::get(op->getContext(), encoding,
                                               lwe::LWEParamsAttr());

    auto encrypt = b.create<cggi::CreateTrivialOp>(lweType, inputValue);

    rewriter.replaceOp(op, encrypt);
    return success();
  }
};

struct ConvertTruncIOp : public OpConversionPattern<mlir::arith::TruncIOp> {
  ConvertTruncIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::TruncIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::TruncIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto outType = convertArithToCGGIType(
        cast<IntegerType>(op.getResult().getType()), op->getContext());
    auto castOp = b.create<cggi::CastOp>(op.getLoc(), outType, adaptor.getIn());

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct ConvertExtUIOp : public OpConversionPattern<mlir::arith::ExtUIOp> {
  ConvertExtUIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto outType = convertArithToCGGIType(
        cast<IntegerType>(op.getResult().getType()), op->getContext());
    auto castOp = b.create<cggi::CastOp>(op.getLoc(), outType, adaptor.getIn());

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct ConvertShRUIOp : public OpConversionPattern<mlir::arith::ShRUIOp> {
  ConvertShRUIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ShRUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ShRUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cteShiftSizeOp = op.getRhs().getDefiningOp<mlir::arith::ConstantOp>();

    if (cteShiftSizeOp) {
      auto outputType = adaptor.getLhs().getType();

      auto shiftAmount = cast<IntegerAttr>(cteShiftSizeOp.getValue())
                             .getValue()
                             .getSExtValue();

      auto inputValue =
          mlir::IntegerAttr::get(rewriter.getI8Type(), (int8_t)shiftAmount);
      auto cteOp = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI8Type(), inputValue);

      auto shiftOp =
          b.create<cggi::ShiftRightOp>(outputType, adaptor.getLhs(), cteOp);
      rewriter.replaceOp(op, shiftOp);

      return success();
    }

    cteShiftSizeOp = op.getLhs().getDefiningOp<mlir::arith::ConstantOp>();

    auto outputType = adaptor.getRhs().getType();

    auto shiftAmount =
        cast<IntegerAttr>(cteShiftSizeOp.getValue()).getValue().getSExtValue();

    auto inputValue = mlir::IntegerAttr::get(rewriter.getI8Type(), shiftAmount);
    auto cteOp = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI8Type(), inputValue);

    auto shiftOp =
        b.create<cggi::ShiftRightOp>(outputType, adaptor.getLhs(), cteOp);
    rewriter.replaceOp(op, shiftOp);
    rewriter.replaceOp(op.getLhs().getDefiningOp(), cteOp);

    return success();
  }
};

struct ArithToCGGI : public impl::ArithToCGGIBase<ArithToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ArithToCGGITypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [](mlir::arith::ConstantOp op) {
          // Allow use of constant if it is used to denote the size of a shift
          bool usedByShift = llvm::any_of(op->getUsers(), [&](Operation *user) {
            return isa<cggi::ShiftRightOp>(user);
          });
          return (isa<IndexType>(op.getValue().getType()) || (usedByShift));
        });

    target.addDynamicallyLegalOp<
        memref::AllocOp, memref::DeallocOp, memref::StoreOp, memref::SubViewOp,
        memref::CopyOp, tensor::FromElementsOp, tensor::ExtractOp,
        affine::AffineStoreOp, affine::AffineLoadOp>([&](Operation *op) {
      return typeConverter.isLegal(op->getOperandTypes()) &&
             typeConverter.isLegal(op->getResultTypes());
    });

    patterns.add<
        ConvertConstantOp, ConvertTruncIOp, ConvertExtUIOp, ConvertShRUIOp,
        ConvertBinOp<mlir::arith::AddIOp, cggi::AddOp>,
        ConvertBinOp<mlir::arith::MulIOp, cggi::MulOp>,
        ConvertBinOp<mlir::arith::SubIOp, cggi::SubOp>,
        ConvertAny<memref::LoadOp>, ConvertAny<memref::AllocOp>,
        ConvertAny<memref::DeallocOp>, ConvertAny<memref::StoreOp>,
        ConvertAny<memref::SubViewOp>, ConvertAny<memref::CopyOp>,
        ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::ExtractOp>,
        ConvertAny<affine::AffineStoreOp>, ConvertAny<affine::AffineLoadOp> >(
        typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::arith
