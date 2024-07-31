#include "lib/Conversion/LWEToPolynomial/LWEToPolynomial.h"

#include <cstddef>
#include <optional>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOPOLYNOMIAL
#include "lib/Conversion/LWEToPolynomial/LWEToPolynomial.h.inc"

class CiphertextTypeConverter : public TypeConverter {
 public:
  // Convert ciphertext to tensor<#dim x !poly.poly<#rings[#level]>>
  // Our precondition is that the key and plaintext have only one
  // dimension, and the ciphertext has two dimensions. (i.e.) The key is a
  // single polynomial and not a tensor of higher dimensions. We may support
  // higher dimensions in the future (for schemes such as TFHE).
  CiphertextTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::RLWECiphertextType type) -> Type {
      auto rlweParams = type.getRlweParams();
      auto ring = rlweParams.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({rlweParams.getDimension()}, polyTy);
    });
    addConversion([ctx](lwe::RLWEPlaintextType type) -> Type {
      auto ring = type.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);
      return polyTy;
    });
    addConversion([ctx](lwe::RLWESecretKeyType type) -> Type {
      auto rlweParams = type.getRlweParams();
      auto ring = rlweParams.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({rlweParams.getDimension() - 1}, polyTy);
    });
    addConversion([ctx](lwe::RLWEPublicKeyType type) -> Type {
      auto rlweParams = type.getRlweParams();
      auto ring = rlweParams.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({rlweParams.getDimension()}, polyTy);
    });
  }
};

struct ConvertRLWEDecrypt : public OpConversionPattern<RLWEDecryptOp> {
  ConvertRLWEDecrypt(mlir::MLIRContext *context)
      : OpConversionPattern<RLWEDecryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto secretKey = adaptor.getSecretKey();

    auto inputDimension = cast<RankedTensorType>(input.getType()).getShape();

    // Note that secretKey dimension needs to be one. The following check is
    // sufficient for this because before this is executed, both the input
    // (a ciphertext) and secretKey (a secret key) are checked to have the same
    // RLWE dimensions; otherwise, it errors out in advance.
    if (inputDimension.size() != 1 || inputDimension.front() != 2) {
      op.emitError() << "Expected 2 dimensional ciphertext, found ciphertext "
                        "tensor dimension = "
                     << inputDimension;
      // TODO: For TFHE, which can support higher dimensional keys, plaintexts,
      // and ciphertexts, we need to add support for encrypt and decrypt for
      // those cases.
      return failure();
    }

    ImplicitLocOpBuilder builder(loc, rewriter);

    // For a ciphertext input = (c_0, c_1), calculates
    // plaintext = secretKey * c_0 + c_1
    auto index0 = builder.create<arith::ConstantIndexOp>(0);
    tensor::ExtractOp extractOp0 =
        builder.create<tensor::ExtractOp>(input, ValueRange{index0});
    auto index1 = builder.create<arith::ConstantIndexOp>(1);
    tensor::ExtractOp extractOp1 =
        builder.create<tensor::ExtractOp>(input, ValueRange{index1});

    tensor::ExtractOp extractSecretKeyOp =
        builder.create<tensor::ExtractOp>(secretKey, ValueRange{index0});
    auto index1sk =
        builder.create<polynomial::MulOp>(extractSecretKeyOp, extractOp0);
    auto plaintext = builder.create<polynomial::AddOp>(index1sk, extractOp1);

    rewriter.replaceOp(op, plaintext);
    return success();
  }
};

struct ConvertRAdd : public OpConversionPattern<RAddOp> {
  ConvertRAdd(mlir::MLIRContext *context)
      : OpConversionPattern<RAddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::polynomial::AddOp>(
        op, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    return success();
  }
};

struct ConvertRSub : public OpConversionPattern<RSubOp> {
  ConvertRSub(mlir::MLIRContext *context)
      : OpConversionPattern<RSubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RSubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::polynomial::SubOp>(
        op, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    return success();
  }
};

struct ConvertRNegate : public OpConversionPattern<RNegateOp> {
  ConvertRNegate(mlir::MLIRContext *context)
      : OpConversionPattern<RNegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RNegateOp op, OpAdaptor adaptor,
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

struct ConvertRMul : public OpConversionPattern<RMulOp> {
  ConvertRMul(mlir::MLIRContext *context)
      : OpConversionPattern<RMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto x = adaptor.getLhs();
    auto xT = cast<RankedTensorType>(x.getType());
    auto y = adaptor.getRhs();
    auto yT = cast<RankedTensorType>(y.getType());

    if (xT.getNumElements() != 2 || yT.getNumElements() != 2) {
      op.emitError() << "`lwe.rmul` expects ciphertext as two polynomials, got "
                     << xT.getNumElements() << " and " << yT.getNumElements();
      return failure();
    }

    if (xT.getElementType() != yT.getElementType()) {
      op->emitOpError() << "`lwe.rmul` expects operands of the same type";
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

struct LWEToPolynomial : public impl::LWEToPolynomialBase<LWEToPolynomial> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    CiphertextTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(context);

    patterns.add<ConvertRLWEDecrypt, ConvertRAdd, ConvertRSub, ConvertRNegate,
                 ConvertRMul>(typeConverter, context);
    target.addIllegalOp<RLWEDecryptOp, RAddOp, RSubOp, RNegateOp, RMulOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);

    // Run full conversion, if any LWE ops were missed out the pass will fail.
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::lwe
