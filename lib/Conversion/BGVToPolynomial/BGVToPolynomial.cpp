#include "lib/Conversion/BGVToPolynomial/BGVToPolynomial.h"

#include <cstddef>
#include <optional>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOPOLYNOMIAL
#include "lib/Conversion/BGVToPolynomial/BGVToPolynomial.h.inc"

class CiphertextTypeConverter : public TypeConverter {
 public:
  // Convert ciphertext to tensor<#dim x !poly.poly<#rings[#level]>>
  CiphertextTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::RLWECiphertextType type) -> Type {
      auto rlweParams = type.getRlweParams();
      auto ring = rlweParams.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({rlweParams.getDimension()}, polyTy);
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
    rewriter.replaceOp(op, rewriter.create<::mlir::polynomial::AddOp>(
                               op.getLoc(), adaptor.getOperands()[0],
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
    rewriter.replaceOp(op, rewriter.create<::mlir::polynomial::SubOp>(
                               op.getLoc(), adaptor.getOperands()[0],
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
    polynomial::PolynomialType polyType = cast<polynomial::PolynomialType>(
        cast<RankedTensorType>(arg.getType()).getElementType());
    auto neg = rewriter.create<arith::ConstantIntOp>(
        loc, -1, polyType.getRing().getCoefficientType());
    rewriter.replaceOp(op, rewriter.create<::mlir::polynomial::MulScalarOp>(
                               loc, arg.getType(), arg, neg));
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
    auto x = adaptor.getLhs();
    auto xT = cast<RankedTensorType>(x.getType());
    auto y = adaptor.getRhs();
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

    auto z0 = b.create<::mlir::polynomial::MulOp>(x0, y0);
    auto x0y1 = b.create<::mlir::polynomial::MulOp>(x0, y1);
    auto x1y0 = b.create<::mlir::polynomial::MulOp>(x1, y0);
    auto z1 = b.create<::mlir::polynomial::AddOp>(x0y1, x1y0);
    auto z2 = b.create<::mlir::polynomial::MulOp>(x1, y1);

    auto z = b.create<tensor::FromElementsOp>(ArrayRef<Value>({z0, z1, z2}));

    rewriter.replaceOp(op, z);
    return success();
  }
};

struct BGVToPolynomial : public impl::BGVToPolynomialBase<BGVToPolynomial> {
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

    addStructuralConversionPatterns(typeConverter, patterns, target);

    // Run full conversion, if any BGV ops were missed out the pass will fail.
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
