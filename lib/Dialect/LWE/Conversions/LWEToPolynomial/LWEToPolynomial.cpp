#include "lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.h"

#include <utility>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Dialect/Random/IR/RandomEnums.h"
#include "lib/Dialect/Random/IR/RandomOps.h"
#include "lib/Dialect/Random/IR/RandomTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
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
#include "lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.h.inc"

class CiphertextTypeConverter : public TypeConverter {
 public:
  // Convert ciphertext to tensor<#dim x !poly.poly<#rings[#level]>>
  // Our precondition is that the key and plaintext have only one
  // dimension, and the ciphertext has two dimensions. (i.e.) The key is a
  // single polynomial and not a tensor of higher dimensions. We may support
  // higher dimensions in the future (for schemes such as TFHE).
  // TODO(#1199): properly lower NewLWEType (often RNS) to PolynomialType.
  CiphertextTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::NewLWECiphertextType type) -> Type {
      auto ring = type.getCiphertextSpace().getRing();
      auto polyTy = ::mlir::heir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({type.getCiphertextSpace().getSize()},
                                   polyTy);
    });
    addConversion([ctx](lwe::NewLWEPlaintextType type) -> Type {
      auto ring = type.getPlaintextSpace().getRing();
      auto polyTy = ::mlir::heir::polynomial::PolynomialType::get(ctx, ring);
      return polyTy;
    });
    addConversion([ctx](lwe::NewLWESecretKeyType type) -> Type {
      auto ring = type.getRing();
      auto polyTy = ::mlir::heir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({2}, polyTy);
    });
    addConversion([ctx](lwe::NewLWEPublicKeyType type) -> Type {
      auto ring = type.getRing();
      auto polyTy = ::mlir::heir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({2}, polyTy);
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
      // TODO (#882): For TFHE, which can support higher dimensional keys,
      // plaintexts, and ciphertexts, we need to add support for encrypt and
      // decrypt for those cases.
      return failure();
    }

    ImplicitLocOpBuilder builder(loc, rewriter);

    // For a ciphertext input = (c_0, c_1), calculates
    // plaintext = secretKey * c_0 + c_1
    auto index0 = arith::ConstantIndexOp::create(builder, 0);
    tensor::ExtractOp extractOp0 =
        tensor::ExtractOp::create(builder, input, ValueRange{index0});
    auto index1 = arith::ConstantIndexOp::create(builder, 1);
    tensor::ExtractOp extractOp1 =
        tensor::ExtractOp::create(builder, input, ValueRange{index1});

    tensor::ExtractOp extractSecretKeyOp =
        tensor::ExtractOp::create(builder, secretKey, ValueRange{index0});
    auto index1sk =
        polynomial::MulOp::create(builder, extractSecretKeyOp, extractOp0);
    auto plaintext = polynomial::AddOp::create(builder, index1sk, extractOp1);

    rewriter.replaceOp(op, plaintext);
    return success();
  }
};

struct ConvertRLWEEncrypt : public OpConversionPattern<RLWEEncryptOp> {
  ConvertRLWEEncrypt(mlir::MLIRContext *context)
      : OpConversionPattern<RLWEEncryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto key = adaptor.getKey();

    // TODO (#785): Migrate to new LWE types and plaintext modulus.
    auto inputT = cast<lwe::RLWEPlaintextType>(op.getInput().getType());
    auto inputEncoding = inputT.getEncoding();
    auto cleartextBitwidthOrFailure =
        llvm::TypeSwitch<Attribute, FailureOr<int>>(inputEncoding)
            .Case<lwe::BitFieldEncodingAttr,
                  lwe::UnspecifiedBitFieldEncodingAttr>(
                [](auto attr) -> FailureOr<int> {
                  return attr.getCleartextBitwidth();
                })
            .Default([](Attribute attr) -> FailureOr<int> {
              llvm_unreachable(
                  "Unsupported encoding attribute for cleartext bitwidth");
              return failure();
            });
    auto cleartextStartOrFailure =
        llvm::TypeSwitch<Attribute, FailureOr<int>>(inputEncoding)
            .Case<lwe::BitFieldEncodingAttr>([](auto attr) -> FailureOr<int> {
              return attr.getCleartextStart();
            })
            .Case<lwe::UnspecifiedBitFieldEncodingAttr>(
                [](auto attr) -> FailureOr<int> {
                  llvm_unreachable(
                      "Upsecified Bit Field Encoding Attribute for cleartext "
                      "start");
                  return failure();
                })
            .Default([](Attribute attr) -> FailureOr<int> {
              llvm_unreachable(
                  "Unsupported encoding attribute for cleartext start");
              return failure();
            });

    // Should return failure if cleartextBitwidth or cleartextStart fail.
    if (failed(cleartextBitwidthOrFailure) || failed(cleartextStartOrFailure)) {
      return failure();
    }
    auto cleartextBitwidth = cleartextBitwidthOrFailure.value();
    auto cleartextStart = cleartextStartOrFailure.value();

    // Check that cleartext_start = cleartext_bitwidth for BGV encryption.
    if (cleartextBitwidth != cleartextStart) {
      // TODO (#882): Add support for other encryption schemes besides BGV. Left
      // as future work.
      op.emitError() << "`lwe.rlwe_encrypt` expects BGV encryption"
                     << " with cleartext_start = cleartext_bitwidth, but"
                     << " found cleartext_start = " << cleartextStart
                     << " and cleartext_bitwidth = " << cleartextBitwidth
                     << ".";
      return failure();
    }

    auto isPublicKey =
        llvm::TypeSwitch<Type, bool>(op.getKey().getType())
            .Case<lwe::NewLWEPublicKeyType>(
                [](auto key) -> bool { return true; })
            .Case<lwe::NewLWESecretKeyType>(
                [](auto key) -> bool { return false; })
            .Default([](Type key) -> bool {
              llvm_unreachable(
                  "Unsupported key type: Neither public key nor secret key");
              return false;
            });

    ImplicitLocOpBuilder builder(loc, rewriter);

    auto index0 = arith::ConstantIndexOp::create(builder, 0);
    auto dimension =
        inputT.getRing().getPolynomialModulus().getPolynomial().getDegree();

    auto coefficientType = inputT.getRing().getCoefficientType();
    auto modArithType = dyn_cast<mod_arith::ModArithType>(coefficientType);
    if (!modArithType) {
      op.emitError() << "Unsupported coefficient type: " << coefficientType;
      return failure();
    }

    Type tensorEltTy = modArithType.getModulus().getType();
    auto tensorParams = RankedTensorType::get({dimension}, tensorEltTy);
    auto modArithTensorType = RankedTensorType::get({dimension}, modArithType);

    // TODO (#881): Add pass options to change the seed (which is currently
    // hardcoded to 0 with index).
    // TODO (#873) : Clean up usage of Random Dialect below using num_bits
    // (currently hardcoded to 32).

    // Initialize random number generator with seed.
    auto generateRandom =
        random::InitOp::create(builder, index0, builder.getI32IntegerAttr(32));

    // Create a uniform discrete random distribution with generated values of
    // -1, 0, 1.
    auto uniformDistributionType = random::DistributionType::get(
        getContext(), random::Distribution::uniform);

    auto uniformDistribution = random::DiscreteUniformDistributionOp::create(
        builder, uniformDistributionType, generateRandom,
        builder.getI32IntegerAttr(-1), builder.getI32IntegerAttr(2));

    // Generate random u polynomial from uniform random ternary distribution
    auto uTensor =
        random::SampleOp::create(builder, tensorParams, uniformDistribution);
    // Convert the tensor of ints to a tensor of mod_arith, then a polynomial
    auto modArithUTensor =
        mod_arith::EncapsulateOp::create(builder, modArithTensorType, uTensor);
    auto u = polynomial::FromTensorOp::create(builder, modArithUTensor,
                                              inputT.getRing());

    // Create a discrete Gaussian distribution
    auto discreteGaussianDistributionType = random::DistributionType::get(
        getContext(), random::Distribution::gaussian);

    auto discreteGaussianDistribution =
        random::DiscreteGaussianDistributionOp::create(
            builder, discreteGaussianDistributionType, generateRandom,
            builder.getI32IntegerAttr(0), builder.getI32IntegerAttr(5));
    // TODO (#881): Add pass options to configure stdev
    // (which is currently hardcoded to 5)

    if (isPublicKey) {
      tensor::ExtractOp publicKey0 =
          tensor::ExtractOp::create(builder, key, ValueRange{index0});
      auto index1 = arith::ConstantIndexOp::create(builder, 1);
      tensor::ExtractOp publicKey1 =
          tensor::ExtractOp::create(builder, key, ValueRange{index1});

      // constantT is 2**(cleartextBitwidth), and is used for scalar
      // multiplication.
      // TODO(#876): Migrate to using the plaintext modulus of the encoding info
      // attributes.
      auto constantT = mod_arith::ConstantOp::create(
          builder, modArithType,
          IntegerAttr::get(modArithType.getModulus().getType(),
                           1 << cleartextBitwidth));

      // generate random e0 polynomial from discrete gaussian distribution
      auto e0Tensor = random::SampleOp::create(builder, tensorParams,
                                               discreteGaussianDistribution);
      auto modArithE0Tensor = mod_arith::EncapsulateOp::create(
          builder, modArithTensorType, e0Tensor);
      auto e0 = polynomial::FromTensorOp::create(builder, modArithE0Tensor,
                                                 inputT.getRing());

      // generate random e1 polynomial from discrete gaussian distribution
      auto e1Tensor = random::SampleOp::create(builder, tensorParams,
                                               discreteGaussianDistribution);
      auto modArithE1Tensor = mod_arith::EncapsulateOp::create(
          builder, modArithTensorType, e1Tensor);
      auto e1 = polynomial::FromTensorOp::create(builder, modArithE1Tensor,
                                                 inputT.getRing());

      // TODO (#882): Other encryption schemes (e.g. CKKS) may multiply the
      // noise or key differently. Add support for those cases.
      // Computing ciphertext0 = publicKey0 * u + e0 *
      // constantT + input
      auto publicKey0U = polynomial::MulOp::create(builder, publicKey0, u);
      auto tE0 = polynomial::MulScalarOp::create(builder, e0, constantT);
      auto pK0UtE0 = polynomial::AddOp::create(builder, publicKey0U, tE0);
      auto ciphertext0 = polynomial::AddOp::create(builder, pK0UtE0, input);

      // Computing ciphertext1 = publicKey1 * u + e1 * constantT
      auto publicKey1U = polynomial::MulOp::create(builder, publicKey1, u);
      auto tE1 = polynomial::MulScalarOp::create(builder, e1, constantT);
      auto ciphertext1 = polynomial::AddOp::create(builder, publicKey1U, tE1);

      // ciphertext = (ciphertext0, ciphertext1)
      auto ciphertext = tensor::FromElementsOp::create(
          builder, llvm::ArrayRef<Value>({ciphertext0, ciphertext1}));
      rewriter.replaceOp(op, ciphertext);
    } else {  // secret key
      // We only support secret key encryption with a single polynomial (typical
      // RLWE parameters, whereas CGGI may use a larger number of polynomials
      // for the secret key).
      if (cast<RankedTensorType>(key.getType()).getNumElements() != 1) {
        return op.emitError()
               << "`lwe.rlwe_encrypt` only supports secret keys with a single "
                  "polynomial, got secret key type "
               << key.getType();
      }

      // Generate random e polynomial from discrete gaussian distribution
      auto eTensor = random::SampleOp::create(builder, tensorParams,
                                              discreteGaussianDistribution);
      auto modArithETensor = mod_arith::EncapsulateOp::create(
          builder, modArithTensorType, eTensor);
      auto e = polynomial::FromTensorOp::create(builder, modArithETensor,
                                                inputT.getRing());

      // TODO (#882): Other encryption schemes (e.g. CKKS) may multiply the
      // noise or key differently. Add support for those cases.
      // ciphertext0 = u
      // Compute ciphertext1 = <u,s> + m + e
      auto keyPoly =
          tensor::ExtractOp::create(builder, key, ValueRange{index0});
      auto us = polynomial::MulOp::create(builder, u, keyPoly);
      auto usM = polynomial::AddOp::create(builder, us, input);
      auto ciphertext1 = polynomial::AddOp::create(builder, usM, e);

      // ciphertext = (u, ciphertext0)
      auto ciphertext = tensor::FromElementsOp::create(
          builder, llvm::ArrayRef<Value>({u, ciphertext1}));
      rewriter.replaceOp(op, ciphertext);
    }

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
    rewriter.replaceOpWithNewOp<::mlir::heir::polynomial::AddOp>(
        op, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    return success();
  }
};

struct ConvertRAddPlain : public OpConversionPattern<RAddPlainOp> {
  ConvertRAddPlain(mlir::MLIRContext *context)
      : OpConversionPattern<RAddPlainOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RAddPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::heir::polynomial::AddOp>(
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
    rewriter.replaceOpWithNewOp<::mlir::heir::polynomial::SubOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertRSubPlain : public OpConversionPattern<RSubPlainOp> {
  ConvertRSubPlain(mlir::MLIRContext *context)
      : OpConversionPattern<RSubPlainOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RSubPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::heir::polynomial::SubOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
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
    FailureOr<Value> neg =
        llvm::TypeSwitch<Type, FailureOr<Value>>(
            polyType.getRing().getCoefficientType())
            .Case<mod_arith::ModArithType>(
                [&](mod_arith::ModArithType type) -> Value {
                  return mod_arith::ConstantOp::create(
                      rewriter, loc, type,
                      IntegerAttr::get(type.getModulus().getType(), -1));
                })
            .Case<IntegerType>([&](IntegerType type) -> Value {
              return arith::ConstantIntOp::create(rewriter, loc, type, -1);
            })
            .Default([&](Type type) -> FailureOr<Value> {
              op.emitError() << "Unsupported coefficient type: " << type;
              return failure();
            });

    if (failed(neg)) {
      return failure();
    }

    rewriter.replaceOp(op, ::mlir::heir::polynomial::MulScalarOp::create(
                               rewriter, loc, arg.getType(), arg, neg.value()));
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
    auto i0 = arith::ConstantIndexOp::create(b, 0);
    auto i1 = arith::ConstantIndexOp::create(b, 1);

    auto x0 =
        tensor::ExtractOp::create(b, xT.getElementType(), x, ValueRange{i0});
    auto x1 =
        tensor::ExtractOp::create(b, xT.getElementType(), x, ValueRange{i1});

    auto y0 =
        tensor::ExtractOp::create(b, yT.getElementType(), y, ValueRange{i0});
    auto y1 =
        tensor::ExtractOp::create(b, yT.getElementType(), y, ValueRange{i1});

    auto z0 = ::mlir::heir::polynomial::MulOp::create(b, x0, y0);
    auto x0y1 = ::mlir::heir::polynomial::MulOp::create(b, x0, y1);
    auto x1y0 = ::mlir::heir::polynomial::MulOp::create(b, x1, y0);
    auto z1 = ::mlir::heir::polynomial::AddOp::create(b, x0y1, x1y0);
    auto z2 = ::mlir::heir::polynomial::MulOp::create(b, x1, y1);

    auto z =
        tensor::FromElementsOp > (ArrayRef < Value::create(b, {z0, z1, z2}));

    rewriter.replaceOp(op, z);
    return success();
  }
};

struct ConvertRMulPlain : public OpConversionPattern<RMulPlainOp> {
  ConvertRMulPlain(mlir::MLIRContext *context)
      : OpConversionPattern<RMulPlainOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RMulPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto x = adaptor.getLhs();
    auto xT = cast<RankedTensorType>(x.getType());
    auto y = adaptor.getRhs();
    auto yT = cast<RankedTensorType>(y.getType());

    if (xT.getNumElements() != 2 || yT.getNumElements() != 1) {
      op.emitError() << "`lwe.rmul_plain` expects ciphertext as two "
                        "polynomials and plaintext as 1, got "
                     << xT.getNumElements() << " and " << yT.getNumElements();
      return failure();
    }

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    // z = mul([x0, x1], [y0]) := [x0y0, x1y0] (Multiply ciphertext [2dim] with
    // plaintext [1dim]), lwe canonicalizes with ciphertext first
    auto repeated =
        tensor::FromElementsOp > (ArrayRef < Value::create(b, {y, y}));
    auto z = polynomial::MulOp::create(b, x, repeated);

    rewriter.replaceOp(op, z);
    return success();
  }
};

struct LWEToPolynomial : public impl::LWEToPolynomialBase<LWEToPolynomial> {
  void runOnOperation() override {
    // TODO(#1199): Remove this emitError once the pass is fixed.
    getOperation()->emitError(
        "LWEToPolynomial conversion pass is broken. See #1199.");
    return;

    MLIRContext *context = &getContext();
    auto *module = getOperation();
    CiphertextTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(context);

    patterns.add<ConvertRLWEDecrypt, ConvertRLWEEncrypt, ConvertRAdd,
                 ConvertRSub, ConvertRNegate, ConvertRMul, ConvertRAddPlain,
                 ConvertRSubPlain, ConvertRMulPlain>(typeConverter, context);
    target.addIllegalOp<RLWEDecryptOp, RLWEEncryptOp, RAddOp, RSubOp, RNegateOp,
                        RMulOp, RAddPlainOp, RSubPlainOp, RMulPlainOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);

    // Run full conversion, if any LWE ops were missed out the pass will fail.
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::lwe
