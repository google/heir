#include "lib/Dialect/Openfhe/Conversions/OpenFHEToEmitC/OpenfheToEmitCDialectInterface.h"

#include <cstdint>
#include <functional>
#include <string>

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::openfhe {

namespace {

template <typename OpType>
struct MemberCallConversion : public OpConversionPattern<OpType> {
  // This callback allows this pattern to be used both for in-place operations
  // (which return void and mutate an operand) and non-in-place operations. The
  // resolver returns the Value that the matched op should be replaced with. For
  // non-in-place ops this is the result value, otherwise it is some operand.
  using ReplacementResolver = std::function<Value(
      OpType, emitc::MemberCallOpaqueOp, typename OpType::Adaptor)>;

  StringRef methodName;
  ReplacementResolver replacementResolver;
  bool isVoidCall;

  MemberCallConversion(TypeConverter& tc, MLIRContext* context,
                       StringRef methodName,
                       ReplacementResolver replacementResolver,
                       bool isVoidCall = false)
      : OpConversionPattern<OpType>(tc, context),
        methodName(methodName),
        replacementResolver(replacementResolver),
        isVoidCall(isVoidCall) {}

  LogicalResult matchAndRewrite(
      OpType op, typename OpType::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value object = adaptor.getOperands().front();
    ValueRange callArgs = adaptor.getOperands().drop_front();

    SmallVector<Type> resultTypes;
    if (!isVoidCall) {
      if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                        resultTypes))) {
        return failure();
      }
    }

    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), resultTypes, object,
        rewriter.getStringAttr(methodName), /*args=*/ArrayAttr(),
        /*template_args=*/ArrayAttr(), callArgs);

    Value replacement = replacementResolver(op, newOp, adaptor);
    if (replacement) {
      rewriter.replaceOp(op, replacement);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct ConvertRotOp : public OpConversionPattern<RotOp> {
  using OpConversionPattern<RotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto cryptoContext = adaptor.getCryptoContext();
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    Value shift;
    if (op.getStaticShiftAttr()) {
      Type shiftType = this->getTypeConverter()->convertType(
          op.getStaticShiftAttr().getType());
      auto constOp = emitc::ConstantOp::create(rewriter, op.getLoc(), shiftType,
                                               op.getStaticShiftAttr());
      shift = constOp.getResult();
    } else if (op.getDynamicShift()) {
      shift = adaptor.getDynamicShift();
    } else {
      return op.emitError(
          "RotOp must have either static_shift or dynamic_shift");
    }

    SmallVector<Value> callArgs = {adaptor.getCiphertext(), shift};
    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, cryptoContext,
        rewriter.getStringAttr("EvalRotate"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertGenParamsOp : public OpConversionPattern<GenParamsOp> {
  using OpConversionPattern<GenParamsOp>::OpConversionPattern;

  void emitMethodCall(ConversionPatternRewriter& rewriter, GenParamsOp op,
                      StringRef methodName, Attribute staticInput) {}

  LogicalResult matchAndRewrite(
      GenParamsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    auto paramsOp = emitc::CallOpaqueOp::create(rewriter, op.getLoc(),
                                                TypeRange{resultType},
                                                "CCParamsT", ValueRange{});
    Value params = paramsOp->getResult(0);

    auto setParam = [&](StringRef methodName, Attribute val) {
      emitc::MemberCallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{},
                                        params,
                                        rewriter.getStringAttr(methodName),
                                        /*args=*/rewriter.getArrayAttr({val}),
                                        /*template_args=*/ArrayAttr(), {});
    };

    if (op.getMulDepth() != 0) {
      setParam("SetMultiplicativeDepth", op.getMulDepthAttr());
    }
    if (op.getPlainMod() != 0) {
      setParam("SetPlaintextModulus", op.getPlainModAttr());
    }
    if (op.getRingDim() != 0) {
      setParam("SetRingDim", op.getRingDimAttr());
    }
    if (op.getBatchSize() != 0) {
      setParam("SetBatchSize", op.getBatchSizeAttr());
    }
    if (op.getFirstModSize() != 0) {
      setParam("SetFirstModSize", op.getFirstModSizeAttr());
    }
    if (op.getScalingModSize() != 0) {
      setParam("SetScalingModSize", op.getScalingModSizeAttr());
    }
    if (op.getEvalAddCount() != 0) {
      setParam("SetEvalAddCount", op.getEvalAddCountAttr());
    }
    if (op.getKeySwitchCount() != 0) {
      setParam("SetKeySwitchCount", op.getKeySwitchCountAttr());
    }
    if (op.getDigitSize() != 0) {
      setParam("SetDigitSize", op.getDigitSizeAttr());
    }
    if (op.getNumLargeDigits() != 0) {
      setParam("SetNumLargeDigits", op.getNumLargeDigitsAttr());
    }
    if (op.getMaxRelinSkDeg() != 0) {
      setParam("SetMaxRelinSkDeg", op.getMaxRelinSkDegAttr());
    }
    if (op.getInsecure()) {
      setParam("SetSecurityLevel",
               rewriter.getStringAttr("lbcrypto::HEStd_NotSet"));
    }
    if (op.getEncryptionTechniqueExtended()) {
      setParam("SetEncryptionTechnique", rewriter.getStringAttr("EXTENDED"));
    }
    if (!op.getKeySwitchingTechniqueBV()) {
      // B/FV defaults to BV, to match HEIR parameter generation we need to
      // set it to HYBRID. Other schemes defaults to HYBRID.
      setParam("SetKeySwitchTechnique", rewriter.getStringAttr("HYBRID"));
    } else {
      setParam("SetKeySwitchTechnique", rewriter.getStringAttr("BV"));
    }
    if (op.getScalingTechniqueFixedManual()) {
      setParam("SetScalingTechnique", rewriter.getStringAttr("FIXEDMANUAL"));
    }

    rewriter.replaceOp(op, params);
    return success();
  }
};

struct ConvertGenContextOp : public OpConversionPattern<GenContextOp> {
  using OpConversionPattern<GenContextOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenContextOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MLIRContext* mlirCtx = op->getContext();
    auto params = adaptor.getParams();
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    auto contextOp = emitc::CallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, "GenCryptoContext",
        ValueRange{params});
    Value context = contextOp->getResult(0);

    auto enableFeature = [&](StringRef featureName) {
      auto featureAttr = emitc::OpaqueAttr::get(mlirCtx, featureName);
      auto staticArgs = rewriter.getArrayAttr({featureAttr});
      emitc::MemberCallOpaqueOp::create(
          rewriter, op.getLoc(), TypeRange{}, context,
          rewriter.getStringAttr("Enable"),
          /*args=*/staticArgs, /*template_args=*/ArrayAttr(), {});
    };

    enableFeature("PKE");
    enableFeature("KEYSWITCH");
    enableFeature("LEVELEDSHE");
    if (op.getSupportFHE()) {
      enableFeature("ADVANCEDSHE");
      enableFeature("FHE");
    }

    rewriter.replaceOp(op, context);
    return success();
  }
};

struct ConvertGenMulKeyOp : public OpConversionPattern<GenMulKeyOp> {
  using OpConversionPattern<GenMulKeyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenMulKeyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto cryptoContext = adaptor.getCryptoContext();
    SmallVector<Value> callArgs = {adaptor.getPrivateKey()};
    emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{}, cryptoContext,
        rewriter.getStringAttr("EvalMultKeyGen"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertGenRotKeyOp : public OpConversionPattern<GenRotKeyOp> {
  using OpConversionPattern<GenRotKeyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenRotKeyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    auto cryptoContext = adaptor.getCryptoContext();
    std::string indicesStr = heir::initializerList(op.getIndices());
    auto indicesConst = emitc::ConstantOp::create(
        rewriter, op.getLoc(),
        TypeRange{emitc::OpaqueType::get(ctx, "std::vector<int32_t>")},
        emitc::OpaqueAttr::get(ctx, indicesStr));

    SmallVector<Value> callArgs = {adaptor.getPrivateKey(),
                                   indicesConst.getResult()};
    emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{}, cryptoContext,
        rewriter.getStringAttr("EvalRotateKeyGen"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertEncryptOp : public OpConversionPattern<EncryptOp> {
  using OpConversionPattern<EncryptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto cryptoContext = adaptor.getCryptoContext();
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    SmallVector<Value> callArgs = {adaptor.getEncryptionKey(),
                                   adaptor.getPlaintext()};
    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, cryptoContext,
        rewriter.getStringAttr("Encrypt"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertDecryptOp : public OpConversionPattern<DecryptOp> {
  using OpConversionPattern<DecryptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    auto cryptoContext = adaptor.getCryptoContext();
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    // Create a destination plaintext to decrypt into
    auto varOp = emitc::VariableOp::create(
        rewriter, op.getLoc(),
        TypeRange{
            emitc::LValueType::get(emitc::OpaqueType::get(ctx, "Plaintext"))},
        emitc::OpaqueAttr::get(ctx, "Plaintext()"));

    auto addrOp = emitc::AddressOfOp::create(
        rewriter, op.getLoc(),
        TypeRange{
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "Plaintext"))},
        varOp.getResult());

    SmallVector<Value> callArgs = {adaptor.getPrivateKey(),
                                   adaptor.getCiphertext(), addrOp.getResult()};
    emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{}, cryptoContext,
        rewriter.getStringAttr("Decrypt"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    auto loadOp = emitc::LoadOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, varOp.getResult());

    rewriter.replaceOp(op, loadOp.getResult());
    return success();
  }
};

struct ConvertDecodeOp : public OpConversionPattern<DecodeOp> {
  using OpConversionPattern<DecodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    auto resultType = op.getResult().getType();
    auto convertedType = this->getTypeConverter()->convertType(resultType);

    auto plaintext = adaptor.getOperands()[0];
    auto vecType = emitc::OpaqueType::get(ctx, "const std::vector<int64_t>&");

    auto getPackedValueOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{vecType}, plaintext,
        rewriter.getStringAttr("GetPackedValue"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), ValueRange{});

    auto dataOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{convertedType},
        getPackedValueOp->getResult(0), rewriter.getStringAttr("data"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), ValueRange{});

    rewriter.replaceOp(op, dataOp);
    return success();
  }
};

struct ConvertMakePackedPlaintextOp
    : public OpConversionPattern<MakePackedPlaintextOp> {
  using OpConversionPattern<MakePackedPlaintextOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MakePackedPlaintextOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    Value val = adaptor.getValue();

    ArrayRef<int64_t> shape =
        cast<ShapedType>(op.getValue().getType()).getShape();
    if (shape.size() != 1) {
      return op.emitOpError() << "expected 1D shape";
    }
    int64_t size = shape[0];

    auto sizeConst = emitc::ConstantOp::create(
        rewriter, op.getLoc(), TypeRange{rewriter.getI64Type()},
        rewriter.getI64IntegerAttr(size));

    // The converted input is a memref type, but OpenFHE takes std::vectors
    // for its plaintexts. So we use the std::vector<int64_t>(first, last)
    // constructor to copy the memref to a vector.
    auto lastPtr =
        emitc::AddOp::create(rewriter, op.getLoc(), TypeRange{val.getType()},
                             val, sizeConst.getResult());
    auto vecOp = emitc::CallOpaqueOp::create(
        rewriter, op.getLoc(),
        TypeRange{emitc::OpaqueType::get(ctx, "std::vector<int64_t>")},
        "std::vector<int64_t>", ValueRange{val, lastPtr.getResult()});

    SmallVector<Value> callArgs = {vecOp->getResult(0)};
    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(),
        TypeRange{emitc::OpaqueType::get(ctx, "Plaintext")},
        adaptor.getCryptoContext(),
        rewriter.getStringAttr("MakePackedPlaintext"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

}  // namespace

void OpenfheToEmitCDialectInterface::populateConvertToEmitCConversionPatterns(
    mlir::ConversionTarget& target, mlir::TypeConverter& typeConverter,
    mlir::RewritePatternSet& patterns, std::optional<bool> lowerToCpp) const {
  auto* context = patterns.getContext();

  typeConverter.addConversion([context, &typeConverter](Type type) -> Type {
    return llvm::TypeSwitch<Type, Type>(type)
        .Case<CryptoContextType>([context](CryptoContextType ty) {
          return emitc::OpaqueType::get(context, "CryptoContextT");
        })
        .Case<CCParamsType>([context](CCParamsType ty) {
          return emitc::OpaqueType::get(context, "CCParamsT");
        })
        .Case<CiphertextType>([context](CiphertextType ty) {
          return emitc::OpaqueType::get(context, "CiphertextT");
        })
        .Case<PlaintextType>([context](PlaintextType ty) {
          return emitc::OpaqueType::get(context, "Plaintext");
        })
        .Case<PublicKeyType>([context](PublicKeyType ty) {
          return emitc::OpaqueType::get(context, "PublicKey");
        })
        .Case<PrivateKeyType>([context](PrivateKeyType ty) {
          return emitc::OpaqueType::get(context, "PrivateKeyT");
        })
        .Case<IndexType>(
            [context](IndexType ty) { return emitc::SizeTType::get(context); })
        .Case<MemRefType>([&typeConverter](MemRefType ty) -> Type {
          auto eltType = ty.getElementType();
          auto convertedEltType = typeConverter.convertType(eltType);
          if (!convertedEltType) return nullptr;
          return emitc::PointerType::get(convertedEltType);
        })
        .Default([](Type ty) { return ty; });
  });

  auto returnsResultResolver = [](auto op, auto newOp, auto adaptor) -> Value {
    return newOp.getResult(0);
  };

  auto inPlaceResolver = [](auto op, auto newOp, auto adaptor) -> Value {
    return adaptor.getOperands()[1];
  };

  patterns.add<MemberCallConversion<AddOp>>(typeConverter, context, "EvalAdd",
                                            returnsResultResolver);
  patterns.add<MemberCallConversion<SubOp>>(typeConverter, context, "EvalSub",
                                            returnsResultResolver);
  patterns.add<MemberCallConversion<MulOp>>(typeConverter, context, "EvalMult",
                                            returnsResultResolver);
  patterns.add<MemberCallConversion<MulNoRelinOp>>(
      typeConverter, context, "EvalMultNoRelin", returnsResultResolver);
  patterns.add<MemberCallConversion<MulPlainOp>>(
      typeConverter, context, "EvalMult", returnsResultResolver);

  patterns.add<MemberCallConversion<AddInPlaceOp>>(
      typeConverter, context, "EvalAddInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<SubInPlaceOp>>(
      typeConverter, context, "EvalSubInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<RelinInPlaceOp>>(
      typeConverter, context, "RelinearizeInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<ModReduceInPlaceOp>>(
      typeConverter, context, "ModReduceInPlace", inPlaceResolver,
      /*isVoidCall=*/true);

  patterns.add<ConvertGenParamsOp, ConvertGenContextOp, ConvertEncryptOp,
               ConvertGenMulKeyOp, ConvertMakePackedPlaintextOp, ConvertRotOp,
               ConvertDecryptOp, ConvertGenRotKeyOp, ConvertDecodeOp>(
      typeConverter, context);

  target.addIllegalDialect<OpenfheDialect>();
}

}  // namespace mlir::heir::openfhe
