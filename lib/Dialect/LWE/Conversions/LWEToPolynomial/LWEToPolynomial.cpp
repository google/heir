#include "lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
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
  }
};

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

    // Cast to the plaintext space types.
    auto plaintextPolyType = cast<polynomial::PolynomialType>(
        typeConverter->convertType(op.getOutput().getType()));
    auto plaintextMod =
        polynomial::ModSwitchOp::create(builder, plaintextPolyType, plaintext);

    rewriter.replaceOp(op, plaintextMod);
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

    auto outputT = cast<lwe::LWECiphertextType>(op.getOutput().getType());

    auto isPublicKey =
        llvm::TypeSwitch<Type, bool>(op.getKey().getType())
            .Case<lwe::LWEPublicKeyType>([](auto key) -> bool { return true; })
            .Case<lwe::LWESecretKeyType>([](auto key) -> bool { return false; })
            .Default([](Type key) -> bool {
              llvm_unreachable(
                  "Unsupported key type: Neither public key nor secret key");
              return false;
            });

    // TODO (#882): Add support for other encryption schemes besides BGV. Left
    // as future work.
    ImplicitLocOpBuilder builder(loc, rewriter);

    auto index0 = arith::ConstantIndexOp::create(builder, 0);

    auto outputTy = cast<RankedTensorType>(
        typeConverter->convertType(op.getOutput().getType()));
    auto outputPolyTy =
        cast<polynomial::PolynomialType>(outputTy.getElementType());

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
    auto u =
        random::SampleOp::create(builder, outputPolyTy, uniformDistribution);

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

      // get plaintext modulus T in the output ring space. We assume the
      // plaintext type is a mod arith type.
      auto plaintextCoeffType =
          outputT.getPlaintextSpace().getRing().getCoefficientType();
      auto plaintextModArithType =
          dyn_cast<mod_arith::ModArithType>(plaintextCoeffType);
      if (!plaintextModArithType) {
        op.emitError() << "Unsupported plaintext coefficient type: "
                       << plaintextCoeffType;
        return failure();
      }

      // create scalar constant T in the output coefficient space
      auto plaintextT = mod_arith::ConstantOp::create(
          builder, plaintextModArithType, plaintextModArithType.getModulus());
      auto constantT = mod_arith::ModSwitchOp::create(
          builder, outputPolyTy.getRing().getCoefficientType(), plaintextT);

      // generate random e0 polynomial from discrete gaussian distribution
      auto e0 = random::SampleOp::create(builder, outputPolyTy,
                                         discreteGaussianDistribution);
      // generate random e1 polynomial from discrete gaussian distribution
      auto e1 = random::SampleOp::create(builder, outputPolyTy,
                                         discreteGaussianDistribution);

      // TODO (#882): Other encryption schemes (e.g. CKKS) may multiply the
      // noise or key differently. Add support for those cases.
      // Computing ciphertext0 = publicKey0 * u + e0 *
      // constantT + cast(input)
      auto publicKey0U = polynomial::MulOp::create(builder, publicKey0, u);
      auto tE0 = polynomial::MulScalarOp::create(builder, e0, constantT);
      auto pK0UtE0 = polynomial::AddOp::create(builder, publicKey0U, tE0);
      // cast from plaintext space to ciphertext space
      auto castInput =
          polynomial::ModSwitchOp::create(builder, outputPolyTy, input);
      auto ciphertext0 = polynomial::AddOp::create(builder, pK0UtE0, castInput);

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
      auto e = random::SampleOp::create(builder, outputPolyTy,
                                        discreteGaussianDistribution);

      // TODO (#882): Other encryption schemes (e.g. CKKS) may multiply the
      // noise or key differently. Add support for those cases.
      // ciphertext0 = u
      // Compute ciphertext1 = <u,s> + m + e
      auto keyPoly =
          tensor::ExtractOp::create(builder, key, ValueRange{index0});
      auto us = polynomial::MulOp::create(builder, u, keyPoly);
      // cast from plaintext space to ciphertext space
      auto castInput =
          polynomial::ModSwitchOp::create(builder, outputPolyTy, input);
      auto usM = polynomial::AddOp::create(builder, us, castInput);
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
  ConvertRAdd(mlir::MLIRContext* context)
      : OpConversionPattern<RAddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<polynomial::AddOp>(op, adaptor.getOperands()[0],
                                                   adaptor.getOperands()[1]);
    return success();
  }
};

struct ConvertRAddPlain : public OpConversionPattern<RAddPlainOp> {
  ConvertRAddPlain(mlir::MLIRContext* context)
      : OpConversionPattern<RAddPlainOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RAddPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<polynomial::AddOp>(op, adaptor.getOperands()[0],
                                                   adaptor.getOperands()[1]);
    return success();
  }
};

struct ConvertRSub : public OpConversionPattern<RSubOp> {
  ConvertRSub(mlir::MLIRContext* context)
      : OpConversionPattern<RSubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RSubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<polynomial::SubOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
    return success();
  }
};

struct ConvertRSubPlain : public OpConversionPattern<RSubPlainOp> {
  ConvertRSubPlain(mlir::MLIRContext* context)
      : OpConversionPattern<RSubPlainOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RSubPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<polynomial::SubOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
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
      return failure();
    }

    rewriter.replaceOp(op, polynomial::MulScalarOp::create(
                               rewriter, loc, arg.getType(), arg, neg.value()));
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
      return failure();
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

struct ConvertRelin : public OpConversionPattern<ckks::RelinearizeOp> {
  ConvertRelin(mlir::MLIRContext* context)
      : OpConversionPattern<ckks::RelinearizeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!op.getKeySwitchingKey()) {
      return failure();
    }

    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value two = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 2);
    Value input0 = rewriter.create<tensor::ExtractOp>(op.getLoc(),
                                                      adaptor.getInput(), zero);
    Value input1 = rewriter.create<tensor::ExtractOp>(op.getLoc(),
                                                      adaptor.getInput(), one);
    Value input2 = rewriter.create<tensor::ExtractOp>(op.getLoc(),
                                                      adaptor.getInput(), two);
    polynomial::KeySwitchInnerOp ksInnerOp =
        rewriter.create<polynomial::KeySwitchInnerOp>(
            op.getLoc(), input2, adaptor.getKeySwitchingKey());

    Value comp0 = rewriter.create<polynomial::AddOp>(
        op.getLoc(), input0, ksInnerOp.getConstantOutput());
    Value comp1 = rewriter.create<polynomial::AddOp>(
        op.getLoc(), input1, ksInnerOp.getLinearOutput());
    Type outputType = RankedTensorType::get(
        {static_cast<int64_t>(2)},
        getElementTypeOrSelf(adaptor.getInput().getType()));
    auto result = rewriter.create<tensor::FromElementsOp>(
        op.getLoc(), outputType, ValueRange{comp0, comp1});
    rewriter.replaceOp(op, result);
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

    patterns.add<ConvertRLWEDecrypt, ConvertRLWEEncrypt, ConvertRAdd,
                 ConvertRSub, ConvertRNegate, ConvertRMul, ConvertRAddPlain,
                 ConvertRSubPlain, ConvertRMulPlain, ConvertRelin>(
        typeConverter, context);
    target.addIllegalOp<RLWEDecryptOp, RLWEEncryptOp, RAddOp, RSubOp, RNegateOp,
                        RMulOp, RAddPlainOp, RSubPlainOp, RMulPlainOp,
                        ckks::RelinearizeOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);

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
