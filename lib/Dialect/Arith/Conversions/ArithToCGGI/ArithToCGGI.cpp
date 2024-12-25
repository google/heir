#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
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

struct ConvertExtSIOp : public OpConversionPattern<mlir::arith::ExtSIOp> {
  ConvertExtSIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtSIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ExtSIOp op, OpAdaptor adaptor,
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
          mlir::IntegerAttr::get(rewriter.getIndexType(), (int8_t)shiftAmount);

      auto shiftOp = b.create<cggi::ScalarShiftRightOp>(
          outputType, adaptor.getLhs(), inputValue);
      rewriter.replaceOp(op, shiftOp);

      return success();
    }

    cteShiftSizeOp = op.getLhs().getDefiningOp<mlir::arith::ConstantOp>();

    auto outputType = adaptor.getRhs().getType();

    auto shiftAmount =
        cast<IntegerAttr>(cteShiftSizeOp.getValue()).getValue().getSExtValue();

    auto inputValue =
        mlir::IntegerAttr::get(rewriter.getIndexType(), shiftAmount);

    auto shiftOp = b.create<cggi::ScalarShiftRightOp>(
        outputType, adaptor.getLhs(), inputValue);
    rewriter.replaceOp(op, shiftOp);

    return success();
  }
};

template <typename SourceArithOp, typename TargetModArithOp>
struct ConvertArithBinOp : public OpConversionPattern<SourceArithOp> {
  ConvertArithBinOp(mlir::MLIRContext *context)
      : OpConversionPattern<SourceArithOp>(context) {}

  using OpConversionPattern<SourceArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceArithOp op, typename SourceArithOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (auto lhsDefOp = op.getLhs().getDefiningOp()) {
      if (isa<mlir::arith::ConstantOp>(lhsDefOp)) {
        auto result = b.create<TargetModArithOp>(adaptor.getRhs().getType(),
                                                 adaptor.getRhs(), op.getLhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    if (auto rhsDefOp = op.getRhs().getDefiningOp()) {
      if (isa<mlir::arith::ConstantOp>(rhsDefOp)) {
        auto result = b.create<TargetModArithOp>(adaptor.getLhs().getType(),
                                                 adaptor.getLhs(), op.getRhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    auto result = b.create<TargetModArithOp>(
        adaptor.getLhs().getType(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
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
    target.addLegalOp<mlir::arith::ConstantOp>();

    target.addDynamicallyLegalOp<mlir::arith::ExtSIOp>([&](Operation *op) {
      if (auto *defOp =
              cast<mlir::arith::ExtSIOp>(op).getOperand().getDefiningOp()) {
        return isa<mlir::arith::ConstantOp>(defOp);
      }
      return false;
    });

    target.addDynamicallyLegalOp<memref::SubViewOp, memref::CopyOp,
                                 tensor::FromElementsOp, tensor::ExtractOp,
                                 affine::AffineStoreOp, affine::AffineLoadOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });

    target.addDynamicallyLegalOp<memref::AllocOp>([&](Operation *op) {
      // Check if all Store ops are constants, if not store op, accepts
      // Check if there is at least one Store op that is a constants
      return (llvm::all_of(op->getUses(),
                           [&](OpOperand &op) {
                             auto defOp =
                                 dyn_cast<memref::StoreOp>(op.getOwner());
                             if (defOp) {
                               return isa<mlir::arith::ConstantOp>(
                                   defOp.getValue().getDefiningOp());
                             }
                             return true;
                           }) &&
              llvm::any_of(op->getUses(),
                           [&](OpOperand &op) {
                             auto defOp =
                                 dyn_cast<memref::StoreOp>(op.getOwner());
                             if (defOp) {
                               return isa<mlir::arith::ConstantOp>(
                                   defOp.getValue().getDefiningOp());
                             }
                             return false;
                           })) ||
             // The other case: Memref need to be in LWE format
             (typeConverter.isLegal(op->getOperandTypes()) &&
              typeConverter.isLegal(op->getResultTypes()));
    });

    target.addDynamicallyLegalOp<memref::StoreOp>([&](Operation *op) {
      if (auto *defOp = cast<memref::StoreOp>(op).getValue().getDefiningOp()) {
        if (isa<mlir::arith::ConstantOp>(defOp)) {
          return true;
        }
      }

      return typeConverter.isLegal(op->getOperandTypes()) &&
             typeConverter.isLegal(op->getResultTypes());
    });

    // Convert LoadOp if memref comes from an argument
    target.addDynamicallyLegalOp<memref::LoadOp>([&](Operation *op) {
      if (typeConverter.isLegal(op->getOperandTypes()) &&
          typeConverter.isLegal(op->getResultTypes())) {
        return true;
      }
      auto loadOp = dyn_cast<memref::LoadOp>(op);

      return loadOp.getMemRef().getDefiningOp() != nullptr;
    });

    patterns.add<
        ConvertTruncIOp, ConvertExtUIOp, ConvertExtSIOp, ConvertShRUIOp,
        ConvertArithBinOp<mlir::arith::AddIOp, cggi::AddOp>,
        ConvertArithBinOp<mlir::arith::MulIOp, cggi::MulOp>,
        ConvertArithBinOp<mlir::arith::SubIOp, cggi::SubOp>,
        ConvertAny<memref::LoadOp>, ConvertAny<memref::AllocOp>,
        ConvertAny<memref::DeallocOp>, ConvertAny<memref::SubViewOp>,
        ConvertAny<memref::CopyOp>, ConvertAny<memref::StoreOp>,
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
