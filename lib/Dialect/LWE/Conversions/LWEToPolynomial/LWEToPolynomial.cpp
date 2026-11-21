#include "lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypeInterfaces.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
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
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOPOLYNOMIAL
#include "lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.h.inc"

RankedTensorType convertCiphertextType(lwe::LWECiphertextType type) {
  auto ring = type.getCiphertextSpace().getRing();
  auto polyTy = polynomial::PolynomialType::get(type.getContext(), ring);
  return RankedTensorType::get({type.getCiphertextSpace().getSize()}, polyTy);
}

class CiphertextTypeConverter : public TypeConverter {
 public:
  // Convert ciphertext to tensor<#dim x !poly.poly<#rings[#level]>>
  // Our precondition is that the key and plaintext have only one
  // dimension, and the ciphertext has two dimensions. (i.e.) The key is a
  // single polynomial and not a tensor of higher dimensions. We may support
  // higher dimensions in the future (for schemes such as TFHE).
  // TODO(#1199): properly lower LWEType (often RNS) to PolynomialType.
  CiphertextTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](lwe::LWECiphertextType type) -> Type {
      return convertCiphertextType(type);
    });
    addConversion([](RankedTensorType type) -> std::optional<Type> {
      auto ctEltTy = dyn_cast<lwe::LWECiphertextType>(type.getElementType());
      if (!ctEltTy) {
        return std::nullopt;
      }
      RankedTensorType convertedEltTy = convertCiphertextType(ctEltTy);

      // If the input type is tensor<d1 x d2 x !ciphertext>, then the result is
      // a tensor<d1 x d2 x N x !poly>>
      SmallVector<int64_t> shape(type.getShape());
      shape.insert(shape.end(), convertedEltTy.getShape().begin(),
                   convertedEltTy.getShape().end());

      return RankedTensorType::get(shape, convertedEltTy.getElementType());
    });
    addConversion([ctx](lwe::LWEPlaintextType type) -> Type {
      auto ring = type.getPlaintextSpace().getRing();
      return polynomial::PolynomialType::get(ctx, ring);
    });
    addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
      auto ring = type.getRing();
      auto polyTy = polynomial::PolynomialType::get(ctx, ring);

      // TODO(#2045): Use size information (number of polynomials) from the LWE
      // key type instead of hardcoding to 1.
      return RankedTensorType::get({1}, polyTy);
    });
    addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
      auto ring = type.getRing();
      auto polyTy = polynomial::PolynomialType::get(ctx, ring);

      // TODO(#2045): Use size information (number of polynomials) from the LWE
      // key type instead of hardcoding to 2.
      return RankedTensorType::get({2}, polyTy);
    });
    addConversion([ctx](lwe::LWERingEltType type) -> Type {
      auto ring = type.getRing();
      return polynomial::PolynomialType::get(ctx, ring);
    });
  }
};

// Returns the plaintext modulus T of `plaintextModulusType`, as a scalar
// constant in the coefficient space of `outputPolyTy`.
static Value getPlaintextModulusInCiphertextSpace(
    ImplicitLocOpBuilder& builder, mod_arith::ModArithType plaintextModulusType,
    polynomial::PolynomialType outputPolyTy) {
  auto plaintextT = mod_arith::ConstantOp::create(
      builder, plaintextModulusType, plaintextModulusType.getModulus());
  return mod_arith::ModSwitchOp::create(
             builder, outputPolyTy.getRing().getCoefficientType(), plaintextT)
      .getResult();
}

// Verifies that `lwe.rlwe_encrypt` and `lwe.rlwe_decrypt` can be lowered for
// the encryption type of `ciphertextType` (see `LweEncryptionType`), and emits
// an error on `op` otherwise. Creates no IR.
static LogicalResult verifySupportedEncryption(
    Operation* op, lwe::LWEPlaintextType plaintextType,
    lwe::LWECiphertextType ciphertextType) {
  lwe::CiphertextSpaceAttr ciphertextSpace =
      ciphertextType.getCiphertextSpace();
  switch (ciphertextSpace.getEncryptionType()) {
    case lwe::LweEncryptionType::lsb:
      return success();
    case lwe::LweEncryptionType::ckks: {
      // A CKKS plaintext is a polynomial holding the encoded (already scaled)
      // message, in the same ring as its ciphertext.
      polynomial::RingAttr plaintextRing =
          plaintextType.getPlaintextSpace().getRing();
      polynomial::RingAttr ciphertextRing = ciphertextSpace.getRing();
      if (!isa<mod_arith::ModQTypeInterface>(
              plaintextRing.getCoefficientType())) {
        return op->emitOpError()
               << "expected plaintext to have modular (e.g., RNS) polynomial "
                  "coefficients, got "
               << plaintextRing.getCoefficientType();
      }
      if (plaintextRing != ciphertextRing) {
        return op->emitOpError() << "expected plaintext and ciphertext to have "
                                    "the same ring, got "
                                 << plaintextRing << " and " << ciphertextRing;
      }
      return success();
    }
    case lwe::LweEncryptionType::msb:
      // TODO(#882): Support `msb` (BFV) encryption and decryption, which
      // scale the message by floor(Q/t).
      return op->emitOpError()
             << "only lsb and ckks encryption are supported, got encryption "
                "type "
             << lwe::stringifyLweEncryptionType(
                    ciphertextSpace.getEncryptionType());
  }
  llvm_unreachable("unknown LweEncryptionType");
}

struct ConvertRLWEDecrypt : public OpConversionPattern<RLWEDecryptOp> {
  ConvertRLWEDecrypt(mlir::MLIRContext* context)
      : OpConversionPattern<RLWEDecryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto secretKey = adaptor.getSecretKey();
    lwe::LWEPlaintextType plaintextType = op.getOutput().getType();
    lwe::LWECiphertextType ciphertextType = op.getInput().getType();

    if (failed(verifySupportedEncryption(op, plaintextType, ciphertextType))) {
      return rewriter.notifyMatchFailure(op, "unsupported encryption");
    }

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
      return rewriter.notifyMatchFailure(op,
                                         "expected 2 dimensional ciphertext");
    }

    ImplicitLocOpBuilder builder(loc, rewriter);

    // A ciphertext (c_0, c_1) represents the polynomial c_0 + c_1 * S in the
    // abstract secret key S (see `lwe.extract_coeff`), so we calculate
    // plaintext = c_0 + c_1 * secretKey
    auto index0 = arith::ConstantIndexOp::create(builder, 0);
    tensor::ExtractOp extractOp0 =
        tensor::ExtractOp::create(builder, input, ValueRange{index0});
    auto index1 = arith::ConstantIndexOp::create(builder, 1);
    tensor::ExtractOp extractOp1 =
        tensor::ExtractOp::create(builder, input, ValueRange{index1});

    tensor::ExtractOp extractSecretKeyOp =
        tensor::ExtractOp::create(builder, secretKey, ValueRange{index0});
    auto c1TimesSk =
        polynomial::MulOp::create(builder, extractOp1, extractSecretKeyOp);
    Value plaintext = polynomial::AddOp::create(builder, extractOp0, c1TimesSk);

    // For BGV this is m + t * e in the ciphertext ring, and casting it to the
    // plaintext ring reduces it modulo t. For CKKS it is m + e, which is
    // already in the plaintext ring; the noise is removed when it is decoded.
    if (ciphertextType.getCiphertextSpace().getEncryptionType() ==
        lwe::LweEncryptionType::lsb) {
      auto plaintextPolyType = cast<polynomial::PolynomialType>(
          typeConverter->convertType(plaintextType));
      plaintext = polynomial::ModSwitchOp::create(builder, plaintextPolyType,
                                                  plaintext);
    }

    rewriter.replaceOp(op, plaintext);
    return success();
  }
};

struct ConvertRLWEEncrypt : public OpConversionPattern<RLWEEncryptOp> {
  ConvertRLWEEncrypt(mlir::MLIRContext* context)
      : OpConversionPattern<RLWEEncryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto key = adaptor.getKey();
    lwe::LWEPlaintextType plaintextType = op.getInput().getType();
    lwe::LWECiphertextType ciphertextType = op.getOutput().getType();
    lwe::LweEncryptionType encryptionType =
        ciphertextType.getCiphertextSpace().getEncryptionType();

    auto isPublicKey =
        llvm::TypeSwitch<Type, bool>(op.getKey().getType())
            .Case<lwe::LWEPublicKeyType>([](auto key) -> bool { return true; })
            .Case<lwe::LWESecretKeyType>([](auto key) -> bool { return false; })
            .Default([](Type key) -> bool {
              llvm_unreachable(
                  "Unsupported key type: Neither public key nor secret key");
              return false;
            });

    // All checks must happen before any IR is created: the pass disables
    // pattern rollback, so IR created by a pattern that then fails is not
    // removed.

    if (failed(verifySupportedEncryption(op, plaintextType, ciphertextType))) {
      return rewriter.notifyMatchFailure(op, "unsupported encryption");
    }

    // BGV scales the encryption noise by the plaintext modulus t, so its
    // plaintext coefficient type must be a mod arith type.
    Type plaintextCoeffType =
        plaintextType.getPlaintextSpace().getRing().getCoefficientType();
    auto plaintextModulusType =
        dyn_cast<mod_arith::ModArithType>(plaintextCoeffType);
    if (encryptionType == lwe::LweEncryptionType::lsb &&
        !plaintextModulusType) {
      op.emitError() << "Unsupported plaintext coefficient type: "
                     << plaintextCoeffType;
      return rewriter.notifyMatchFailure(
          op, "unsupported plaintext coefficient type");
    }

    auto outputTy =
        cast<RankedTensorType>(typeConverter->convertType(ciphertextType));
    auto outputPolyTy =
        cast<polynomial::PolynomialType>(outputTy.getElementType());

    if (!isPublicKey) {
      // We only support secret key encryption with a single polynomial (typical
      // RLWE parameters, whereas CGGI may use a larger number of polynomials
      // for the secret key).
      if (cast<RankedTensorType>(key.getType()).getNumElements() != 1) {
        op.emitError()
            << "`lwe.rlwe_encrypt` only supports secret keys with a single "
               "polynomial, got secret key type "
            << key.getType();
        return rewriter.notifyMatchFailure(
            op, "secret key has more than one polynomial");
      }

      // The mask `a` is sampled uniformly from R_Q, which requires a modular
      // coefficient type and an explicit ring modulus.
      if (!isa<mod_arith::ModQTypeInterface>(
              outputPolyTy.getRing().getCoefficientType()) ||
          !outputPolyTy.getRing().getPolynomialModulus()) {
        op.emitError() << "Unsupported ciphertext ring: "
                       << outputPolyTy.getRing();
        return rewriter.notifyMatchFailure(op, "unsupported ciphertext ring");
      }
    }

    ImplicitLocOpBuilder builder(loc, rewriter);

    auto index0 = arith::ConstantIndexOp::create(builder, 0);

    // The ciphertext is built so that c_0 + c_1 * s decrypts to
    //   lsb (BGV): m + t * e, with m cast from the plaintext ring into R_Q
    //   ckks:      m + e,     with m already in R_Q
    Value message = input;
    Value noiseScale;  // null: the noise is not scaled
    if (encryptionType == lwe::LweEncryptionType::lsb) {
      message = polynomial::ModSwitchOp::create(builder, outputPolyTy, input);
      noiseScale = getPlaintextModulusInCiphertextSpace(
          builder, plaintextModulusType, outputPolyTy);
    }
    auto scaleNoise = [&](Value e) -> Value {
      if (!noiseScale) return e;
      return polynomial::MulScalarOp::create(builder, e, noiseScale);
    };

    // TODO (#881): Add pass options to change the seed (which is currently
    // hardcoded to 0 with index).
    // TODO (#873) : Clean up usage of Random Dialect below using num_bits
    // (currently hardcoded to 32).

    // Initialize random number generator with seed.
    auto generateRandom =
        random::InitOp::create(builder, index0, builder.getI32IntegerAttr(32));

    auto uniformDistributionType = random::DistributionType::get(
        getContext(), random::Distribution::uniform);

    // Create a discrete Gaussian distribution
    auto discreteGaussianDistributionType = random::DistributionType::get(
        getContext(), random::Distribution::gaussian);

    auto discreteGaussianDistribution =
        random::DiscreteGaussianDistributionOp::create(
            builder, discreteGaussianDistributionType, generateRandom,
            builder.getI32IntegerAttr(0), builder.getI32IntegerAttr(5));
    // TODO (#881): Add pass options to configure stdev
    // (which is currently hardcoded to 5)

    Value ciphertext;
    if (isPublicKey) {
      // Create a uniform discrete random distribution with generated values of
      // -1, 0, 1.
      auto uniformDistribution = random::DiscreteUniformDistributionOp::create(
          builder, uniformDistributionType, generateRandom,
          builder.getI32IntegerAttr(-1), builder.getI32IntegerAttr(2));

      // Generate random u polynomial from uniform random ternary distribution
      auto u =
          random::SampleOp::create(builder, outputPolyTy, uniformDistribution);

      tensor::ExtractOp publicKey0 =
          tensor::ExtractOp::create(builder, key, ValueRange{index0});
      auto index1 = arith::ConstantIndexOp::create(builder, 1);
      tensor::ExtractOp publicKey1 =
          tensor::ExtractOp::create(builder, key, ValueRange{index1});

      // generate random e0 polynomial from discrete gaussian distribution
      auto e0 = random::SampleOp::create(builder, outputPolyTy,
                                         discreteGaussianDistribution);
      // generate random e1 polynomial from discrete gaussian distribution
      auto e1 = random::SampleOp::create(builder, outputPolyTy,
                                         discreteGaussianDistribution);

      // Computing
      //   ciphertext0 = publicKey0 * u + scaleNoise(e0) + message
      //   ciphertext1 = publicKey1 * u + scaleNoise(e1)
      auto publicKey0U = polynomial::MulOp::create(builder, publicKey0, u);
      Value scaledE0 = scaleNoise(e0);
      auto pK0UPlusE0 =
          polynomial::AddOp::create(builder, publicKey0U, scaledE0);
      auto ciphertext0 =
          polynomial::AddOp::create(builder, pK0UPlusE0, message);

      auto publicKey1U = polynomial::MulOp::create(builder, publicKey1, u);
      Value scaledE1 = scaleNoise(e1);
      auto ciphertext1 =
          polynomial::AddOp::create(builder, publicKey1U, scaledE1);

      // ciphertext = (ciphertext0, ciphertext1)
      ciphertext = tensor::FromElementsOp::create(
          builder, llvm::ArrayRef<Value>({ciphertext0, ciphertext1}));
    } else {  // secret key
      // The mask `a` must be sampled uniformly at random from R_Q, so we
      // sample each of its coefficients uniformly modulo Q. For an RNS
      // ciphertext modulus Q, this samples each limb uniformly.
      Type ciphertextCoeffType = outputPolyTy.getRing().getCoefficientType();
      polynomial::IntPolynomialAttr polyModulus =
          outputPolyTy.getRing().getPolynomialModulus();
      auto coeffsType = RankedTensorType::get(
          {polyModulus.getPolynomial().getDegree()}, ciphertextCoeffType);
      auto aCoeffs = random::SampleModQOp::create(builder, coeffsType,
                                                  generateRandom.getResult());
      auto a = polynomial::FromTensorOp::create(builder, outputPolyTy,
                                                aCoeffs.getResult());

      // Generate random e polynomial from discrete gaussian distribution
      auto e = random::SampleOp::create(builder, outputPolyTy,
                                        discreteGaussianDistribution);

      // Computing
      //   ciphertext0 = -a * s + scaleNoise(e) + message
      //   ciphertext1 = a
      // so that ciphertext0 + ciphertext1 * s = message + scaleNoise(e),
      // matching the lowering of `lwe.rlwe_decrypt`.
      auto keyPoly =
          tensor::ExtractOp::create(builder, key, ValueRange{index0});
      auto aTimesS = polynomial::MulOp::create(builder, a, keyPoly);
      Value scaledE = scaleNoise(e);
      auto mPlusE = polynomial::AddOp::create(builder, message, scaledE);
      auto ciphertext0 = polynomial::SubOp::create(builder, mPlusE, aTimesS);

      // ciphertext = (ciphertext0, a)
      ciphertext = tensor::FromElementsOp::create(
          builder, llvm::ArrayRef<Value>({ciphertext0, a}));
    }

    rewriter.replaceOp(op, ciphertext);
    return success();
  }
};

struct ConvertRNegate : public OpConversionPattern<RNegateOp> {
  ConvertRNegate(mlir::MLIRContext* context)
      : OpConversionPattern<RNegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RNegateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
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
      return rewriter.notifyMatchFailure(op, "unsupported coefficient type");
    }

    rewriter.replaceOp(op, polynomial::MulScalarOp::create(
                               rewriter, loc, arg.getType(), arg, neg.value()));
    return success();
  }
};

struct ConvertMulScalar : public OpConversionPattern<MulScalarOp> {
  ConvertMulScalar(mlir::MLIRContext* context)
      : OpConversionPattern<MulScalarOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulScalarOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getOutput().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "Result type could not be converted");
    }
    rewriter.replaceOpWithNewOp<polynomial::MulScalarOp>(
        op, resultType, adaptor.getCiphertext(), adaptor.getScalar());
    return success();
  }
};

struct ConvertRMul : public OpConversionPattern<RMulOp> {
  ConvertRMul(mlir::MLIRContext* context)
      : OpConversionPattern<RMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto x = adaptor.getLhs();
    auto xT = cast<RankedTensorType>(x.getType());
    auto y = adaptor.getRhs();
    auto yT = cast<RankedTensorType>(y.getType());

    if (xT.getNumElements() != 2 || yT.getNumElements() != 2) {
      op.emitError() << "`lwe.rmul` expects ciphertext as two polynomials, got "
                     << xT.getNumElements() << " and " << yT.getNumElements();
      return rewriter.notifyMatchFailure(op, "ciphertext not two polynomials");
    }

    if (xT.getElementType() != yT.getElementType()) {
      return rewriter.notifyMatchFailure(op, "operands not of same type");
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

    auto z0 = polynomial::MulOp::create(b, x0, y0);
    auto x0y1 = polynomial::MulOp::create(b, x0, y1);
    auto x1y0 = polynomial::MulOp::create(b, x1, y0);
    auto z1 = polynomial::AddOp::create(b, x0y1, x1y0);
    auto z2 = polynomial::MulOp::create(b, x1, y1);

    auto z = tensor::FromElementsOp::create(b, ArrayRef<Value>{z0, z1, z2});

    rewriter.replaceOp(op, z);
    return success();
  }
};

struct ConvertRMulPlain : public OpConversionPattern<RMulPlainOp> {
  ConvertRMulPlain(mlir::MLIRContext* context)
      : OpConversionPattern<RMulPlainOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RMulPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto x = adaptor.getLhs();
    auto xT = cast<RankedTensorType>(x.getType());
    auto y = adaptor.getRhs();
    auto yT = cast<RankedTensorType>(y.getType());

    if (xT.getNumElements() != 2 || yT.getNumElements() != 1) {
      op.emitError() << "`lwe.rmul_plain` expects ciphertext as two "
                        "polynomials and plaintext as 1, got "
                     << xT.getNumElements() << " and " << yT.getNumElements();
      return rewriter.notifyMatchFailure(
          op, "ciphertext not two polynomials or plaintext not one polynomial");
    }

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    // z = mul([x0, x1], [y0]) := [x0y0, x1y0] (Multiply ciphertext [2dim] with
    // plaintext [1dim]), lwe canonicalizes with ciphertext first
    auto repeated = tensor::FromElementsOp::create(b, ArrayRef<Value>{y, y});
    auto z = polynomial::MulOp::create(b, x, repeated);

    rewriter.replaceOp(op, z);
    return success();
  }
};

struct ConvertRMulRingElt : public OpConversionPattern<RMulRingEltOp> {
  ConvertRMulRingElt(mlir::MLIRContext* context)
      : OpConversionPattern<RMulRingEltOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RMulRingEltOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto x = adaptor.getLhs();
    auto y = adaptor.getRhs();
    auto yT = dyn_cast<RankedTensorType>(y.getType());
    if (!yT) {
      op.emitError() << "yT is not a RankedTensorType";
    }

    if (yT.getNumElements() != 2) {
      op.emitError() << "`lwe.rmul_ring_elt` expects ciphertext to be two "
                        "polynomials; got "
                     << yT.getNumElements();
      return rewriter.notifyMatchFailure(
          op,
          "ciphertext not two polynomials or ring element not one polynomial");
    }

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    // z = mul([x0], [y0, y1]) := [x0y0, x0y1] (Multiply ciphertext [2dim] with
    // plaintext [1dim])
    auto repeatedX = tensor::FromElementsOp::create(b, ArrayRef<Value>{x, x});
    auto z = polynomial::MulOp::create(b, repeatedX, y);

    rewriter.replaceOp(op, z);
    return success();
  }
};

struct ConvertExtractCoeff : public OpConversionPattern<ExtractCoeffOp> {
  ConvertExtractCoeff(mlir::MLIRContext* context)
      : OpConversionPattern<ExtractCoeffOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractCoeffOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    int64_t indexInt = op.getIndex().getZExtValue();
    Value indexValue = arith::ConstantIndexOp::create(b, indexInt);
    Value polyOut =
        tensor::ExtractOp::create(b, adaptor.getValue(), indexValue);
    rewriter.replaceOp(op, polyOut);
    return success();
  }
};

struct ConvertFromCoeffs : public OpConversionPattern<FromCoeffsOp> {
  ConvertFromCoeffs(mlir::MLIRContext* context)
      : OpConversionPattern<FromCoeffsOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromCoeffsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    ValueRange convertedCoeffs = adaptor.getCoeffs();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "Result type could not be converted");
    }
    Value outputTensor =
        tensor::FromElementsOp::create(b, resultType, convertedCoeffs);
    rewriter.replaceOp(op, outputTensor);
    return success();
  }
};

struct ConvertExtractSlice : public OpConversionPattern<ExtractSliceOp> {
  ConvertExtractSlice(mlir::MLIRContext* context)
      : OpConversionPattern<ExtractSliceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value polySlice = polynomial::ExtractSliceOp::create(
        b, adaptor.getInput(), adaptor.getStart(), adaptor.getSize());
    rewriter.replaceOp(op, polySlice);
    return success();
  }
};

struct ConvertConvertBasis : public OpConversionPattern<ConvertBasisOp> {
  ConvertConvertBasis(mlir::MLIRContext* context)
      : OpConversionPattern<ConvertBasisOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConvertBasisOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    MLIRContext* context = op.getContext();

    polynomial::PolynomialType inputPolyTy =
        cast<polynomial::PolynomialType>(adaptor.getValue().getType());
    polynomial::RingAttr ringAttr = inputPolyTy.getRing();
    rns::RNSType targetBasisTy = dyn_cast<rns::RNSType>(op.getTargetBasis());
    if (!targetBasisTy) return failure();
    polynomial::RingAttr outputRingAttr = polynomial::RingAttr::get(
        context, targetBasisTy, ringAttr.getPolynomialModulus());
    polynomial::PolynomialType outputPolyTy =
        polynomial::PolynomialType::get(context, outputRingAttr);

    Value coeffLoop = polynomial::ApplyCoefficientwiseOp::create(
        b, adaptor.getValue(), outputPolyTy,
        [&](ImplicitLocOpBuilder& nestedBuilder, Value coeff,
            Value _) -> Value {
          return rns::ConvertBasisOp::create(nestedBuilder, coeff,
                                             targetBasisTy);
        });
    rewriter.replaceOp(op, coeffLoop);
    return success();
  }
};

struct LWEToPolynomial : public impl::LWEToPolynomialBase<LWEToPolynomial> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    CiphertextTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(context);

    patterns.add<ConvertRLWEDecrypt, ConvertRLWEEncrypt,
                 ConvertBinOp<RAddOp, polynomial::AddOp>,
                 ConvertBinOp<RSubOp, polynomial::SubOp>, ConvertRNegate,
                 ConvertMulScalar, ConvertRMul,
                 ConvertBinOp<RAddPlainOp, polynomial::AddOp>,
                 ConvertBinOp<RSubPlainOp, polynomial::SubOp>, ConvertRMulPlain,
                 ConvertRMulRingElt, ConvertExtractCoeff, ConvertFromCoeffs,
                 ConvertExtractSlice, ConvertConvertBasis>(typeConverter,
                                                           context);
    target.addIllegalOp<RLWEDecryptOp, RLWEEncryptOp, RAddOp, RSubOp, RNegateOp,
                        MulScalarOp, RMulOp, RMulRingEltOp, RAddPlainOp,
                        RSubPlainOp, RMulPlainOp, ExtractCoeffOp, FromCoeffsOp,
                        ExtractSliceOp, ConvertBasisOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);
    // This helps handle tensors of ciphertexts,
    // which get lowered to multi-dimensional tensors of polynomials
    addTensorOfTensorConversionPatterns(typeConverter, patterns, target);

    // Run full conversion, if any LWE ops were missed out the pass will fail.
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(
            applyFullConversion(module, target, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::lwe
