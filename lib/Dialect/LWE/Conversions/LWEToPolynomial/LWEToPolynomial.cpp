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

    auto index0 = builder.create<arith::ConstantIndexOp>(0);
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
        builder.create<random::InitOp>(index0, builder.getI32IntegerAttr(32));

    // Create a uniform discrete random distribution with generated values of
    // -1, 0, 1.
    auto uniformDistributionType = random::DistributionType::get(
        getContext(), random::Distribution::uniform);

    auto uniformDistribution =
        builder.create<random::DiscreteUniformDistributionOp>(
            uniformDistributionType, generateRandom,
            builder.getI32IntegerAttr(-1), builder.getI32IntegerAttr(2));

    // Generate random u polynomial from uniform random ternary distribution
    auto uTensor =
        builder.create<random::SampleOp>(tensorParams, uniformDistribution);
    // Convert the tensor of ints to a tensor of mod_arith, then a polynomial
    auto modArithUTensor =
        builder.create<mod_arith::EncapsulateOp>(modArithTensorType, uTensor);
    auto u = builder.create<polynomial::FromTensorOp>(modArithUTensor,
                                                      inputT.getRing());

    // Create a discrete Gaussian distribution
    auto discreteGaussianDistributionType = random::DistributionType::get(
        getContext(), random::Distribution::gaussian);

    auto discreteGaussianDistribution =
        builder.create<random::DiscreteGaussianDistributionOp>(
            discreteGaussianDistributionType, generateRandom,
            builder.getI32IntegerAttr(0), builder.getI32IntegerAttr(5));
    // TODO (#881): Add pass options to configure stdev
    // (which is currently hardcoded to 5)

    if (isPublicKey) {
      tensor::ExtractOp publicKey0 =
          builder.create<tensor::ExtractOp>(key, ValueRange{index0});
      auto index1 = builder.create<arith::ConstantIndexOp>(1);
      tensor::ExtractOp publicKey1 =
          builder.create<tensor::ExtractOp>(key, ValueRange{index1});

      // constantT is 2**(cleartextBitwidth), and is used for scalar
      // multiplication.
      // TODO(#876): Migrate to using the plaintext modulus of the encoding info
      // attributes.
      auto constantT = builder.create<mod_arith::ConstantOp>(
          modArithType, IntegerAttr::get(modArithType.getModulus().getType(),
                                         1 << cleartextBitwidth));

      // generate random e0 polynomial from discrete gaussian distribution
      auto e0Tensor = builder.create<random::SampleOp>(
          tensorParams, discreteGaussianDistribution);
      auto modArithE0Tensor = builder.create<mod_arith::EncapsulateOp>(
          modArithTensorType, e0Tensor);
      auto e0 = builder.create<polynomial::FromTensorOp>(modArithE0Tensor,
                                                         inputT.getRing());

      // generate random e1 polynomial from discrete gaussian distribution
      auto e1Tensor = builder.create<random::SampleOp>(
          tensorParams, discreteGaussianDistribution);
      auto modArithE1Tensor = builder.create<mod_arith::EncapsulateOp>(
          modArithTensorType, e1Tensor);
      auto e1 = builder.create<polynomial::FromTensorOp>(modArithE1Tensor,
                                                         inputT.getRing());

      // TODO (#882): Other encryption schemes (e.g. CKKS) may multiply the
      // noise or key differently. Add support for those cases.
      // Computing ciphertext0 = publicKey0 * u + e0 *
      // constantT + input
      auto publicKey0U = builder.create<polynomial::MulOp>(publicKey0, u);
      auto tE0 = builder.create<polynomial::MulScalarOp>(e0, constantT);
      auto pK0UtE0 = builder.create<polynomial::AddOp>(publicKey0U, tE0);
      auto ciphertext0 = builder.create<polynomial::AddOp>(pK0UtE0, input);

      // Computing ciphertext1 = publicKey1 * u + e1 * constantT
      auto publicKey1U = builder.create<polynomial::MulOp>(publicKey1, u);
      auto tE1 = builder.create<polynomial::MulScalarOp>(e1, constantT);
      auto ciphertext1 = builder.create<polynomial::AddOp>(publicKey1U, tE1);

      // ciphertext = (ciphertext0, ciphertext1)
      auto ciphertext = builder.create<tensor::FromElementsOp>(
          llvm::ArrayRef<Value>({ciphertext0, ciphertext1}));
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
      auto eTensor = builder.create<random::SampleOp>(
          tensorParams, discreteGaussianDistribution);
      auto modArithETensor =
          builder.create<mod_arith::EncapsulateOp>(modArithTensorType, eTensor);
      auto e = builder.create<polynomial::FromTensorOp>(modArithETensor,
                                                        inputT.getRing());

      // TODO (#882): Other encryption schemes (e.g. CKKS) may multiply the
      // noise or key differently. Add support for those cases.
      // ciphertext0 = u
      // Compute ciphertext1 = <u,s> + m + e
      auto keyPoly = builder.create<tensor::ExtractOp>(key, ValueRange{index0});
      auto us = builder.create<polynomial::MulOp>(u, keyPoly);
      auto usM = builder.create<polynomial::AddOp>(us, input);
      auto ciphertext1 = builder.create<polynomial::AddOp>(usM, e);

      // ciphertext = (u, ciphertext0)
      auto ciphertext = builder.create<tensor::FromElementsOp>(
          llvm::ArrayRef<Value>({u, ciphertext1}));
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
                  return rewriter.create<mod_arith::ConstantOp>(
                      loc, type,
                      IntegerAttr::get(type.getModulus().getType(), -1));
                })
            .Case<IntegerType>([&](IntegerType type) -> Value {
              return rewriter.create<arith::ConstantIntOp>(loc, type, -1);
            })
            .Default([&](Type type) -> FailureOr<Value> {
              op.emitError() << "Unsupported coefficient type: " << type;
              return failure();
            });

    if (failed(neg)) {
      return failure();
    }

    rewriter.replaceOp(op,
                       rewriter.create<::mlir::heir::polynomial::MulScalarOp>(
                           loc, arg.getType(), arg, neg.value()));
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

    auto z0 = b.create<::mlir::heir::polynomial::MulOp>(x0, y0);
    auto x0y1 = b.create<::mlir::heir::polynomial::MulOp>(x0, y1);
    auto x1y0 = b.create<::mlir::heir::polynomial::MulOp>(x1, y0);
    auto z1 = b.create<::mlir::heir::polynomial::AddOp>(x0y1, x1y0);
    auto z2 = b.create<::mlir::heir::polynomial::MulOp>(x1, y1);

    auto z = b.create<tensor::FromElementsOp>(ArrayRef<Value>({z0, z1, z2}));

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
    auto repeated = b.create<tensor::FromElementsOp>(ArrayRef<Value>({y, y}));
    auto z = b.create<polynomial::MulOp>(x, repeated);

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
