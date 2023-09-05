#include "include/Conversion/BGVToPoly/BGVToPoly.h"

#include <cstddef>
#include <optional>

#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/BGV/IR/BGVOps.h"
#include "include/Dialect/BGV/IR/BGVTypes.h"
#include "include/Dialect/Poly/IR/PolyAttributes.h"
#include "include/Dialect/Poly/IR/PolyOps.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "include/Dialect/Poly/IR/Polynomial.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOPOLY
#include "include/Conversion/BGVToPoly/BGVToPoly.h.inc"

class CiphertextTypeConverter : public TypeConverter {
 public:
  // Convert ciphertext to tensor<#dim x !poly.poly<#rings[#level]>>
  CiphertextTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](CiphertextType type) -> Type {
      assert(type.getLevel().has_value());
      auto level = type.getLevel().value();
      assert(level < type.getRings().getRings().size());

      auto ring = type.getRings().getRings()[level];
      auto polyTy = poly::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({type.getDim()}, polyTy);
    });
  }
  // We don't include any custom materialization ops because this lowering is
  // all done in a single pass. The dialect conversion framework works by
  // resolving intermediate (mid-pass) type conflicts by inserting
  // unrealized_conversion_cast ops, and only converting those to custom
  // materializations if they persist at the end of the pass. In our case,
  // we'd only need to use custom materializations if we split this lowering
  // across multiple passes.
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, rewriter.create<poly::AddOp>(op.getLoc(), adaptor.getOperands()[0],
                                         adaptor.getOperands()[1]));
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  ConvertSub(mlir::MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, rewriter.create<poly::SubOp>(op.getLoc(), adaptor.getOperands()[0],
                                         adaptor.getOperands()[1]));
    return success();
  }
};

struct ConvertNegate : public OpConversionPattern<Negate> {
  ConvertNegate(mlir::MLIRContext *context)
      : OpConversionPattern<Negate>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Negate op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto arg = adaptor.getOperands()[0];
    auto neg = rewriter.create<arith::ConstantIntOp>(loc, -1, /*width=*/8);
    rewriter.replaceOp(
        op, rewriter.create<poly::MulConstantOp>(loc, arg.getType(), arg, neg));
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
    auto x = adaptor.getX();
    auto xT = cast<RankedTensorType>(x.getType());
    auto y = adaptor.getY();
    auto yT = cast<RankedTensorType>(y.getType());

    if (xT.getNumElements() != 2 || yT.getNumElements() != 2) {
      op.emitError() << "`bgv.mul` expects ciphertext as two polynomials, got "
                     << xT.getNumElements() << " and " << yT.getNumElements();
      return failure();
    }

    if (xT.getElementType() != yT.getElementType()) {
      op->emitOpError() << "`bgv.mul` expects operands of the same type";
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // z = mul([x0, x1], [y0, y1]) := [x0.y0, x0.y1 + x1.y0, x1.y1]
    auto i0 = b.create<arith::ConstantIndexOp>(0);
    auto i1 = b.create<arith::ConstantIndexOp>(1);

    auto x0 =
        b.create<tensor::ExtractOp>(xT.getElementType(), x, ValueRange{i0});
    auto x1 =
        b.create<tensor::ExtractOp>(xT.getElementType(), x, ValueRange{i1});

    auto y0 =
        b.create<tensor::ExtractOp>(yT.getElementType(), y, ValueRange{i0});
    auto y1 =
        b.create<tensor::ExtractOp>(yT.getElementType(), y, ValueRange{i1});

    auto z0 = b.create<poly::MulOp>(x0, y0);
    auto x0y1 = b.create<poly::MulOp>(x0, y1);
    auto x1y0 = b.create<poly::MulOp>(x1, y0);
    auto z1 = b.create<poly::AddOp>(x0y1, x1y0);
    auto z2 = b.create<poly::MulOp>(x1, y1);

    auto z = b.create<tensor::FromElementsOp>(ArrayRef<Value>({z0, z1, z2}));

    rewriter.replaceOp(op, z);
    return success();
  }
};

struct BGVToPoly : public impl::BGVToPolyBase<BGVToPoly> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    CiphertextTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(context);

    patterns.add<ConvertAdd, ConvertSub, ConvertNegate, ConvertMul>(
        typeConverter, context);
    target.addIllegalOp<AddOp, SubOp, Negate, MulOp>();

    // Add "standard" set of conversions and constraints for full dialect
    // conversion. See Dialect/Func/Transforms/FuncBufferize.cpp for example.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    // Run full conversion, if any BGV ops were missed out the pass will fail.
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
