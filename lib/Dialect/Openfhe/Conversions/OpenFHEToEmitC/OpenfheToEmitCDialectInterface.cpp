#include "lib/Dialect/Openfhe/Conversions/OpenFHEToEmitC/OpenfheToEmitCDialectInterface.h"

#include <cstdint>
#include <functional>
#include <string>

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"       // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::openfhe {

namespace {

Value derefIfSmartPtr(Value val, OpBuilder& builder, Location loc) {
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(val.getType())) {
    if (opaqueTy.getValue() == "CryptoContextT") {
      auto ctx = builder.getContext();
      auto implTy = emitc::OpaqueType::get(
          ctx, "lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>&");
      return builder.create<emitc::CallOpaqueOp>(loc, implTy, "*", val)
          .getResult(0);
    }
    if (opaqueTy.getValue() == "Plaintext" ||
        opaqueTy.getValue() == "PlaintextT") {
      auto ctx = builder.getContext();
      auto implTy = emitc::OpaqueType::get(ctx, "lbcrypto::PlaintextImpl&");
      return builder.create<emitc::CallOpaqueOp>(loc, implTy, "*", val)
          .getResult(0);
    }
  }
  return val;
}

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
    Value object =
        derefIfSmartPtr(adaptor.getOperands().front(), rewriter, op.getLoc());
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

struct ConvertLevelReduceOp : public OpConversionPattern<LevelReduceOp> {
  using OpConversionPattern<LevelReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LevelReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    auto nullKeyOp = emitc::ConstantOp::create(
        rewriter, op.getLoc(),
        TypeRange{emitc::OpaqueType::get(ctx, "EvalKeyT")},
        emitc::OpaqueAttr::get(ctx, "nullptr"));

    auto levelConst = emitc::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(op.getLevelToDrop()));

    SmallVector<Value> callArgs = {
        adaptor.getCiphertext(), nullKeyOp.getResult(), levelConst.getResult()};
    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, cryptoContext,
        rewriter.getStringAttr("LevelReduce"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertLevelReduceInPlaceOp
    : public OpConversionPattern<LevelReduceInPlaceOp> {
  using OpConversionPattern<LevelReduceInPlaceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LevelReduceInPlaceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());

    auto nullKeyOp = emitc::ConstantOp::create(
        rewriter, op.getLoc(),
        TypeRange{emitc::OpaqueType::get(ctx, "EvalKeyT")},
        emitc::OpaqueAttr::get(ctx, "nullptr"));

    auto levelConst = emitc::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(op.getLevelToDrop()));

    SmallVector<Value> callArgs = {
        adaptor.getCiphertext(), nullKeyOp.getResult(), levelConst.getResult()};

    emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{}, cryptoContext,
        rewriter.getStringAttr("LevelReduceInPlace"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, adaptor.getCiphertext());
    return success();
  }
};

struct ConvertRotOp : public OpConversionPattern<RotOp> {
  using OpConversionPattern<RotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
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
      if (auto stringAttr = dyn_cast<StringAttr>(val)) {
        val = emitc::OpaqueAttr::get(rewriter.getContext(),
                                     stringAttr.getValue());
      }
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
    Value derefedContext = derefIfSmartPtr(context, rewriter, op.getLoc());

    auto enableFeature = [&](StringRef featureName) {
      auto featureAttr = emitc::OpaqueAttr::get(mlirCtx, featureName);
      auto staticArgs = rewriter.getArrayAttr({featureAttr});
      emitc::MemberCallOpaqueOp::create(
          rewriter, op.getLoc(), TypeRange{}, derefedContext,
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

struct ConvertSetupBootstrapOp : public OpConversionPattern<SetupBootstrapOp> {
  using OpConversionPattern<SetupBootstrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SetupBootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, loc);

    uint32_t encodeBudget = op.getLevelBudgetEncode().getValue().getZExtValue();
    uint32_t decodeBudget = op.getLevelBudgetDecode().getValue().getZExtValue();

    std::string initStr = "std::vector<uint32_t>{" +
                          std::to_string(encodeBudget) + ", " +
                          std::to_string(decodeBudget) + "}";

    auto vecType = emitc::OpaqueType::get(ctx, "std::vector<uint32_t>");
    auto varOp = emitc::VariableOp::create(
        rewriter, loc, TypeRange{emitc::LValueType::get(vecType)},
        emitc::OpaqueAttr::get(ctx, initStr));

    SmallVector<Value> callArgs = {varOp.getResult()};
    emitc::MemberCallOpaqueOp::create(
        rewriter, loc, TypeRange{}, cryptoContext,
        rewriter.getStringAttr("EvalBootstrapSetup"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertGenMulKeyOp : public OpConversionPattern<GenMulKeyOp> {
  using OpConversionPattern<GenMulKeyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenMulKeyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
    SmallVector<Value> callArgs = {adaptor.getPrivateKey()};
    emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{}, cryptoContext,
        rewriter.getStringAttr("EvalMultKeyGen"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertGenBootstrapKeyOp
    : public OpConversionPattern<GenBootstrapKeyOp> {
  using OpConversionPattern<GenBootstrapKeyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenBootstrapKeyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, loc);
    auto privateKey = adaptor.getPrivateKey();

    // slots = GetRingDimension() / 2
    auto N_op = emitc::MemberCallOpaqueOp::create(
        rewriter, loc, TypeRange{rewriter.getI64Type()}, cryptoContext,
        rewriter.getStringAttr("GetRingDimension"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), ValueRange{});
    Value N = N_op.getResult(0);

    auto twoAttr = rewriter.getI64IntegerAttr(2);
    auto twoOp = emitc::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                           twoAttr);
    Value two = twoOp.getResult();

    auto slotsOp =
        emitc::DivOp::create(rewriter, loc, rewriter.getI64Type(), N, two);
    Value slots = slotsOp.getResult();

    SmallVector<Value> callArgs = {privateKey, slots};
    emitc::MemberCallOpaqueOp::create(
        rewriter, loc, TypeRange{}, cryptoContext,
        rewriter.getStringAttr("EvalBootstrapKeyGen"),
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
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
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
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
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
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
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

    auto plaintext =
        derefIfSmartPtr(adaptor.getOperands()[0], rewriter, op.getLoc());
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
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc()),
        rewriter.getStringAttr("MakePackedPlaintext"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertMakeCKKSPackedPlaintextOp
    : public OpConversionPattern<MakeCKKSPackedPlaintextOp> {
  using OpConversionPattern<MakeCKKSPackedPlaintextOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MakeCKKSPackedPlaintextOp op, OpAdaptor adaptor,
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

    auto lastPtr =
        emitc::AddOp::create(rewriter, op.getLoc(), TypeRange{val.getType()},
                             val, sizeConst.getResult());
    auto vecOp = emitc::CallOpaqueOp::create(
        rewriter, op.getLoc(),
        TypeRange{emitc::OpaqueType::get(ctx, "std::vector<double>")},
        "std::vector<double>", ValueRange{val, lastPtr.getResult()});

    SmallVector<Value> callArgs = {vecOp->getResult(0)};
    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(),
        TypeRange{emitc::OpaqueType::get(ctx, "Plaintext")},
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc()),
        rewriter.getStringAttr("MakeCKKSPackedPlaintext"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertFastRotationPrecomputeOp
    : public OpConversionPattern<FastRotationPrecomputeOp> {
  using OpConversionPattern<FastRotationPrecomputeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FastRotationPrecomputeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    SmallVector<Value> callArgs = {adaptor.getInput()};
    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, cryptoContext,
        rewriter.getStringAttr("EvalFastRotationPrecompute"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertFastRotationOp : public OpConversionPattern<FastRotationOp> {
  using OpConversionPattern<FastRotationOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FastRotationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto cryptoContext =
        derefIfSmartPtr(adaptor.getCryptoContext(), rewriter, op.getLoc());
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    Value index = adaptor.getIndex();
    auto cyclotomicOrderAttr = op.getCyclotomicOrderAttr();
    Value cyclotomicOrder;
    if (cyclotomicOrderAttr.getInt() == 0) {
      auto N_op = emitc::MemberCallOpaqueOp::create(
          rewriter, op.getLoc(), TypeRange{rewriter.getI64Type()},
          cryptoContext, rewriter.getStringAttr("GetRingDimension"),
          /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), ValueRange{});
      Value N = N_op.getResult(0);

      auto twoAttr = rewriter.getI64IntegerAttr(2);
      auto twoOp = emitc::ConstantOp::create(rewriter, op.getLoc(),
                                             rewriter.getI64Type(), twoAttr);
      Value two = twoOp.getResult();

      auto mOp = emitc::MulOp::create(rewriter, op.getLoc(),
                                      rewriter.getI64Type(), N, two);
      cyclotomicOrder = mOp.getResult();
    } else {
      auto i64Attr = rewriter.getI64IntegerAttr(cyclotomicOrderAttr.getInt());
      auto constOp = emitc::ConstantOp::create(rewriter, op.getLoc(),
                                               rewriter.getI64Type(), i64Attr);
      cyclotomicOrder = constOp.getResult();
    }

    SmallVector<Value> callArgs = {adaptor.getInput(), index, cyclotomicOrder,
                                   adaptor.getPrecomputedDigitDecomp()};

    auto newOp = emitc::MemberCallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, cryptoContext,
        rewriter.getStringAttr("EvalFastRotation"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), callArgs);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertCopy : public OpConversionPattern<memref::CopyOp> {
  ConvertCopy(TypeConverter& tc, MLIRContext* context)
      : OpConversionPattern<memref::CopyOp>(tc, context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto src = adaptor.getSource();
    auto target = adaptor.getTarget();

    MemRefType srcMemrefType = cast<MemRefType>(op.getSource().getType());
    Type convertedElementType =
        this->getTypeConverter()->convertType(srcMemrefType.getElementType());
    if (!convertedElementType) {
      return rewriter.notifyMatchFailure(
          loc, "failed to convert memref element type");
    }

    IndexType indexType = rewriter.getIndexType();
    int64_t numElements = llvm::product_of(srcMemrefType.getShape());

    if (numElements == 1) {
      auto c0Attr = rewriter.getIndexAttr(0);
      auto c0Op = emitc::ConstantOp::create(rewriter, loc, indexType, c0Attr);
      Value c0 = c0Op.getResult();

      auto typedSrc = llvm::cast<TypedValue<emitc::PointerType>>(src);
      auto typedTarget = llvm::cast<TypedValue<emitc::PointerType>>(target);

      auto srcSubscript =
          emitc::SubscriptOp::create(rewriter, loc, typedSrc, c0);
      auto srcLoad = emitc::LoadOp::create(rewriter, loc, convertedElementType,
                                           srcSubscript);

      auto targetSubscript =
          emitc::SubscriptOp::create(rewriter, loc, typedTarget, c0);

      auto assignOp =
          emitc::AssignOp::create(rewriter, loc, targetSubscript, srcLoad);
      rewriter.replaceOp(op, assignOp);
      return success();
    }

    // Calculate total size in bytes
    emitc::CallOpaqueOp elementSize = emitc::CallOpaqueOp::create(
        rewriter, loc, emitc::SizeTType::get(rewriter.getContext()),
        rewriter.getStringAttr("sizeof"), ValueRange{},
        ArrayAttr::get(rewriter.getContext(),
                       {TypeAttr::get(convertedElementType)}));

    emitc::ConstantOp numElementsValue = emitc::ConstantOp::create(
        rewriter, loc, indexType, rewriter.getIndexAttr(numElements));

    Type sizeTType = emitc::SizeTType::get(rewriter.getContext());
    emitc::MulOp totalSizeBytes = emitc::MulOp::create(
        rewriter, loc, sizeTType, elementSize.getResult(0), numElementsValue);

    // Emit memcpy
    auto memcpyCall = emitc::CallOpaqueOp::create(
        rewriter, loc, TypeRange{}, "memcpy",
        ValueRange{target, src, totalSizeBytes.getResult()});

    rewriter.replaceOp(op, memcpyCall.getResults());
    return success();
  }
};

struct ConvertDecodeCKKSOp : public OpConversionPattern<DecodeCKKSOp> {
  using OpConversionPattern<DecodeCKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DecodeCKKSOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = op.getContext();
    Location loc = op.getLoc();
    auto tensorTy = cast<RankedTensorType>(op.getResult().getType());

    // 1. Get size
    int64_t size = tensorTy.getNumElements();

    // 2. Convert element type
    auto convertedEltType =
        this->getTypeConverter()->convertType(tensorTy.getElementType());
    if (!convertedEltType) {
      return rewriter.notifyMatchFailure(loc, "failed to convert element type");
    }

    // 3. Calculate total size in bytes (size * sizeof(ElementType))
    emitc::CallOpaqueOp elementSize = emitc::CallOpaqueOp::create(
        rewriter, loc, emitc::SizeTType::get(ctx),
        rewriter.getStringAttr("sizeof"), ValueRange{},
        ArrayAttr::get(ctx, {TypeAttr::get(convertedEltType)}));

    IndexType indexType = rewriter.getIndexType();
    emitc::ConstantOp numElementsValue = emitc::ConstantOp::create(
        rewriter, loc, indexType, rewriter.getIndexAttr(size));

    Type sizeTType = emitc::SizeTType::get(ctx);
    emitc::MulOp totalSizeBytes = emitc::MulOp::create(
        rewriter, loc, sizeTType, elementSize.getResult(0), numElementsValue);

    // 4. Call malloc
    auto mallocCall = emitc::CallOpaqueOp::create(
        rewriter, loc,
        TypeRange{emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))},
        "malloc", ValueRange{totalSizeBytes.getResult()});
    auto castOp = emitc::CastOp::create(
        rewriter, loc, emitc::PointerType::get(convertedEltType),
        mallocCall.getResult(0));
    Value ptr = castOp.getResult();

    // 5. Call GetCKKSPackedValue
    auto vecType =
        emitc::OpaqueType::get(ctx, "const std::vector<std::complex<double>>&");
    auto getPackedValueOp = emitc::MemberCallOpaqueOp::create(
        rewriter, loc, TypeRange{vecType},
        derefIfSmartPtr(adaptor.getInput(), rewriter, loc),
        rewriter.getStringAttr("GetCKKSPackedValue"),
        /*args=*/ArrayAttr(), /*template_args=*/ArrayAttr(), ValueRange{});

    // 6. Emit std::transform
    std::string formatStr =
        "std::transform(std::begin({}), std::begin({}) + " +
        std::to_string(size) +
        ", {}, [](const std::complex<double>& c) {{ return c.real(); });";
    emitc::VerbatimOp::create(rewriter, loc, formatStr,
                              ValueRange{getPackedValueOp.getResult(0),
                                         getPackedValueOp.getResult(0), ptr});

    // 7. Replace op with pointer
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

static std::string denseElementsToCppInitializer(DenseElementsAttr attr) {
  std::string result = "{";
  Type eltType = attr.getElementType();
  bool first = true;
  if (llvm::isa<IntegerType, IndexType>(eltType)) {
    for (const APInt& val : attr.getValues<APInt>()) {
      if (!first) result += ", ";
      result += std::to_string(val.getSExtValue());
      first = false;
    }
  } else if (llvm::isa<FloatType>(eltType)) {
    for (const APFloat& val : attr.getValues<APFloat>()) {
      if (!first) result += ", ";
      SmallVector<char> str;
      val.toString(str);
      result += std::string(str.begin(), str.end());
      first = false;
    }
  } else {
    assert(false && "unsupported element type for EmitC initializer");
  }
  result += "}";
  return result;
}

struct ConvertGlobal : public OpConversionPattern<memref::GlobalOp> {
  ConvertGlobal(TypeConverter& tc, MLIRContext* context)
      : OpConversionPattern<memref::GlobalOp>(tc, context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      memref::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MemRefType memrefType = op.getType();
    auto ctx = op.getContext();
    Location loc = op.getLoc();

    if (!memrefType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "cannot transform global with dynamic shape");
    }

    auto convertedEltType =
        this->getTypeConverter()->convertType(memrefType.getElementType());
    if (!convertedEltType) {
      return rewriter.notifyMatchFailure(loc, "failed to convert element type");
    }

    Type resultTy;
    if (memrefType.getRank() == 0) {
      resultTy = convertedEltType;
    } else {
      resultTy =
          emitc::ArrayType::get(ctx, memrefType.getShape(), convertedEltType);
    }

    SymbolTable::Visibility visibility = SymbolTable::getSymbolVisibility(op);
    bool staticSpecifier = visibility == SymbolTable::Visibility::Private;
    bool externSpecifier = !staticSpecifier;

    Attribute initialValue = adaptor.getInitialValueAttr();
    if (initialValue) {
      if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(initialValue)) {
        std::string cppInit = denseElementsToCppInitializer(denseAttr);
        initialValue = emitc::OpaqueAttr::get(ctx, cppInit);
      }
    }
    if (memrefType.getRank() == 0) {
      if (initialValue) {
        if (auto elementsAttr = llvm::dyn_cast<ElementsAttr>(initialValue)) {
          initialValue = elementsAttr.getSplatValue<Attribute>();
        }
      }
    }
    if (isa_and_present<UnitAttr>(initialValue)) initialValue = {};

    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, adaptor.getSymName(), resultTy, initialValue, externSpecifier,
        staticSpecifier, adaptor.getConstant());
    return success();
  }
};

struct ConvertGetGlobal : public OpConversionPattern<memref::GetGlobalOp> {
  ConvertGetGlobal(TypeConverter& tc, MLIRContext* context)
      : OpConversionPattern<memref::GetGlobalOp>(tc, context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      memref::GetGlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = op.getType();
    auto ctx = op.getContext();

    auto convertedEltType =
        this->getTypeConverter()->convertType(memrefType.getElementType());
    if (!convertedEltType) {
      return rewriter.notifyMatchFailure(loc, "failed to convert element type");
    }

    if (memrefType.getRank() == 0) {
      auto lvalueType = emitc::LValueType::get(convertedEltType);
      auto globalLValue = emitc::GetGlobalOp::create(rewriter, loc, lvalueType,
                                                     adaptor.getNameAttr());
      auto pointerType = emitc::PointerType::get(convertedEltType);
      auto ptr =
          emitc::AddressOfOp::create(rewriter, loc, pointerType, globalLValue);
      rewriter.replaceOp(op, ptr.getResult());
      return success();
    }

    auto arrayType =
        emitc::ArrayType::get(ctx, memrefType.getShape(), convertedEltType);
    auto globalOp = emitc::GetGlobalOp::create(rewriter, loc, arrayType,
                                               adaptor.getNameAttr());

    auto globalArrayValue =
        cast<TypedValue<emitc::ArrayType>>(globalOp.getResult());

    emitc::ConstantOp zeroIndex = emitc::ConstantOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    llvm::SmallVector<Value> indices(memrefType.getRank(),
                                     zeroIndex.getResult());
    emitc::SubscriptOp subPtr = emitc::SubscriptOp::create(
        rewriter, loc, globalArrayValue, ValueRange(indices));
    emitc::AddressOfOp ptr = emitc::AddressOfOp::create(
        rewriter, loc, emitc::PointerType::get(convertedEltType), subPtr);

    rewriter.replaceOp(op, ptr.getResult());
    return success();
  }
};

struct ConvertToTensorOp
    : public OpConversionPattern<bufferization::ToTensorOp> {
  using OpConversionPattern<bufferization::ToTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      bufferization::ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertToBufferOp
    : public OpConversionPattern<bufferization::ToBufferOp> {
  using OpConversionPattern<bufferization::ToBufferOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      bufferization::ToBufferOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

static Value computeRowMajorLinearIndex(OpBuilder& builder, Location loc,
                                        MemRefType memrefType,
                                        ValueRange indices) {
  ArrayRef<int64_t> shape = memrefType.getShape();
  Type idxType =
      indices.empty() ? builder.getIndexType() : indices[0].getType();

  Value linearIndex = indices.empty()
                          ? builder.create<emitc::ConstantOp>(
                                loc, idxType, builder.getIndexAttr(0))
                          : indices[0];

  for (auto [dim, idx] : llvm::zip(shape.drop_front(), indices.drop_front())) {
    Value dimSize = builder.create<emitc::ConstantOp>(
        loc, idxType, builder.getIndexAttr(dim));
    linearIndex =
        builder.create<emitc::MulOp>(loc, idxType, linearIndex, dimSize);
    linearIndex = builder.create<emitc::AddOp>(loc, idxType, linearIndex, idx);
  }
  return linearIndex;
}

struct ConvertFloorDivSIOp : public OpConversionPattern<arith::FloorDivSIOp> {
  using OpConversionPattern<arith::FloorDivSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::FloorDivSIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resultTy = typeConverter->convertType(op.getType());
    if (!resultTy) return failure();
    rewriter.replaceOpWithNewOp<emitc::DivOp>(op, resultTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ConvertMemRefCast : public OpConversionPattern<memref::CastOp> {
  using OpConversionPattern<memref::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

struct ConvertMemRefExpandShape
    : public OpConversionPattern<memref::ExpandShapeOp> {
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::ExpandShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertMemRefCollapseShape
    : public OpConversionPattern<memref::CollapseShapeOp> {
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::CollapseShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertMemRefSubView : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    MemRefType resultType = op.getType();
    MemRefType sourceType = op.getSourceType();

    int64_t sourceOffset;
    SmallVector<int64_t, 4> sourceStrides;
    if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset))) {
      return rewriter.notifyMatchFailure(op, "failed to get source strides");
    }
    for (int64_t stride : sourceStrides) {
      if (ShapedType::isDynamic(stride)) {
        return rewriter.notifyMatchFailure(op, "dynamic strides not supported");
      }
    }

    Value source = adaptor.getSource();
    auto mixedOffsets = op.getMixedOffsets();
    unsigned dynamicIdx = 0;
    SmallVector<Value> termValues;

    for (unsigned i = 0; i < mixedOffsets.size(); ++i) {
      OpFoldResult ofr = mixedOffsets[i];
      Value offsetVal;
      if (ofr.is<Value>()) {
        offsetVal = adaptor.getOffsets()[dynamicIdx++];
      } else {
        auto attr = ofr.get<Attribute>();
        auto intAttr = llvm::cast<IntegerAttr>(attr);
        int64_t staticOffset = intAttr.getInt();
        offsetVal = rewriter.create<emitc::ConstantOp>(
            loc, rewriter.getIndexType(), rewriter.getIndexAttr(staticOffset));
      }

      int64_t stride = sourceStrides[i];
      if (stride == 0) {
        continue;
      }
      if (stride == 1) {
        termValues.push_back(offsetVal);
      } else {
        Value strideVal = rewriter.create<emitc::ConstantOp>(
            loc, offsetVal.getType(), rewriter.getIndexAttr(stride));
        Value term = rewriter.create<emitc::MulOp>(loc, offsetVal.getType(),
                                                   offsetVal, strideVal);
        termValues.push_back(term);
      }
    }

    if (termValues.empty()) {
      rewriter.replaceOp(op, source);
      return success();
    }

    Value linearOffset = termValues[0];
    for (size_t i = 1; i < termValues.size(); ++i) {
      linearOffset = rewriter.create<emitc::AddOp>(loc, linearOffset.getType(),
                                                   linearOffset, termValues[i]);
    }

    auto typedPtr = llvm::cast<TypedValue<emitc::PointerType>>(source);
    auto subscript =
        emitc::SubscriptOp::create(rewriter, loc, typedPtr, linearOffset);

    Type convertedEltType =
        typeConverter->convertType(resultType.getElementType());
    if (!convertedEltType) return failure();

    auto addr = rewriter.create<emitc::AddressOfOp>(
        loc, emitc::PointerType::get(convertedEltType), subscript);

    rewriter.replaceOp(op, addr.getResult());
    return success();
  }
};

struct ConvertExtFOp : public OpConversionPattern<arith::ExtFOp> {
  using OpConversionPattern<arith::ExtFOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ExtFOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type srcType = op.getIn().getType();
    if (!llvm::isa<TensorType, MemRefType>(srcType)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto shapedType = llvm::cast<ShapedType>(srcType);
    int64_t numElements = shapedType.getNumElements();

    auto resultType = op.getType();
    auto convertedResultType = typeConverter->convertType(resultType);
    if (!convertedResultType) return failure();

    auto resultEltType = llvm::cast<ShapedType>(resultType).getElementType();
    auto convertedResultEltType = typeConverter->convertType(resultEltType);

    Value sizeOfElt =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, rewriter.getType<emitc::SizeTType>(), "sizeof",
                ValueRange(),
                rewriter.getArrayAttr({TypeAttr::get(convertedResultEltType)}))
            .getResult(0);

    Value numEltsConst = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(numElements));

    Value totalSize = rewriter.create<emitc::MulOp>(
        loc, rewriter.getType<emitc::SizeTType>(), sizeOfElt, numEltsConst);

    Value alignment = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getType<emitc::SizeTType>(), rewriter.getIndexAttr(64));

    Value rawPtr = rewriter
                       .create<emitc::CallOpaqueOp>(
                           loc,
                           emitc::PointerType::get(emitc::OpaqueType::get(
                               rewriter.getContext(), "void")),
                           "aligned_alloc", ValueRange({alignment, totalSize}))
                       .getResult(0);

    Value typedPtr =
        rewriter.create<emitc::CastOp>(loc, convertedResultType, rawPtr);

    std::string srcEltName =
        llvm::cast<ShapedType>(srcType).getElementType().isF32() ? "float"
                                                                 : "double";
    std::string dstEltName = resultEltType.isF32() ? "float" : "double";

    std::string transformCode = "std::transform({}, {} + {}, {}, [](" +
                                srcEltName + " x) {{ return (" + dstEltName +
                                ")x; });";

    Value srcVal = adaptor.getIn();
    rewriter.create<emitc::VerbatimOp>(
        loc, transformCode,
        ValueRange({srcVal, srcVal, numEltsConst, typedPtr}));

    rewriter.replaceOp(op, typedPtr);
    return success();
  }
};

struct ConvertModule : public OpConversionPattern<ModuleOp> {
  using OpConversionPattern<ModuleOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModuleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Block* body = op.getBody();
    if (!body->empty()) {
      if (auto verbatim = dyn_cast<emitc::VerbatimOp>(body->front())) {
        if (verbatim.getValue().contains("#include <cstddef>")) {
          return failure();
        }
      }
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(body);

    std::string headers =
        "#include <cstddef>\n"
        "#include <cstdint>\n"
        "#include <cstdlib>\n"
        "#include <cstring>\n"
        "#include <algorithm>\n"
        "#include \"openfhe.h\"\n"
        "using namespace lbcrypto;\n"
        "using CiphertextT = Ciphertext<DCRTPoly>;\n"
        "using ConstCiphertextT = ConstCiphertext<DCRTPoly>;\n"
        "using CCParamsT = CCParams<CryptoContextCKKSRNS>;\n"
        "using CryptoContextT = CryptoContext<DCRTPoly>;\n"
        "using EvalKeyT = EvalKey<DCRTPoly>;\n"
        "using PlaintextT = Plaintext;\n"
        "using PrivateKeyT = PrivateKey<DCRTPoly>;\n"
        "using PublicKeyT = PublicKey<DCRTPoly>;\n"
        "using FastRotPrecompT = std::shared_ptr<std::vector<DCRTPoly>>;\n"
        "#pragma clang diagnostic ignored \"-Wmissing-braces\"\n"
        "\n"
        "template<typename T>\n"
        "inline T* heir_alloc(size_t size) {\n"
        "  return new T[size];\n"
        "}\n"
        "\n"
        "template<typename T>\n"
        "inline void heir_free(T* ptr) {\n"
        "  delete[] ptr;\n"
        "}\n";

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), headers);

    return success();
  }
};

struct CustomLoadOpConversion : public OpConversionPattern<memref::LoadOp> {
  CustomLoadOpConversion(TypeConverter& tc, MLIRContext* context)
      : OpConversionPattern<memref::LoadOp>(tc, context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto resultTy = this->getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(loc, "cannot convert result type");
    }

    Value memref = adaptor.getMemref();
    Type memrefType = memref.getType();

    if (auto arrayType = llvm::dyn_cast<emitc::ArrayType>(memrefType)) {
      auto typedArray = llvm::cast<TypedValue<emitc::ArrayType>>(memref);
      auto subscript = emitc::SubscriptOp::create(rewriter, loc, typedArray,
                                                  adaptor.getIndices());
      rewriter.replaceOpWithNewOp<emitc::LoadOp>(op, resultTy, subscript);
      return success();
    }

    if (auto ptrType = llvm::dyn_cast<emitc::PointerType>(memrefType)) {
      MemRefType opMemrefType =
          llvm::cast<MemRefType>(op.getMemref().getType());
      Value linearIndex = computeRowMajorLinearIndex(
          rewriter, loc, opMemrefType, adaptor.getIndices());
      auto typedPtr = llvm::cast<TypedValue<emitc::PointerType>>(memref);
      auto subscript =
          emitc::SubscriptOp::create(rewriter, loc, typedPtr, linearIndex);
      rewriter.replaceOpWithNewOp<emitc::LoadOp>(op, resultTy, subscript);
      return success();
    }

    return rewriter.notifyMatchFailure(loc, "expected array or pointer type");
  }
};

struct CustomStoreOpConversion : public OpConversionPattern<memref::StoreOp> {
  CustomStoreOpConversion(TypeConverter& tc, MLIRContext* context)
      : OpConversionPattern<memref::StoreOp>(tc, context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value memref = adaptor.getMemref();
    Type memrefType = memref.getType();
    Value value = adaptor.getValue();

    if (auto arrayType = llvm::dyn_cast<emitc::ArrayType>(memrefType)) {
      auto typedArray = llvm::cast<TypedValue<emitc::ArrayType>>(memref);
      auto subscript = emitc::SubscriptOp::create(rewriter, loc, typedArray,
                                                  adaptor.getIndices());
      rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscript, value);
      return success();
    }

    if (auto ptrType = llvm::dyn_cast<emitc::PointerType>(memrefType)) {
      MemRefType opMemrefType =
          llvm::cast<MemRefType>(op.getMemref().getType());
      Value linearIndex = computeRowMajorLinearIndex(
          rewriter, loc, opMemrefType, adaptor.getIndices());
      auto typedPtr = llvm::cast<TypedValue<emitc::PointerType>>(memref);
      auto subscript =
          emitc::SubscriptOp::create(rewriter, loc, typedPtr, linearIndex);
      rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscript, value);
      return success();
    }

    return rewriter.notifyMatchFailure(loc, "expected array or pointer type");
  }
};

struct CustomCallOpConversion : public OpConversionPattern<func::CallOp> {
  CustomCallOpConversion(TypeConverter& tc, MLIRContext* context)
      : OpConversionPattern<func::CallOp>(tc, context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      func::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.getNumResults() > 1) {
      return failure();
    }

    SmallVector<Type> convertedResultTypes;
    for (Type t : op.getResultTypes()) {
      Type resultType = this->getTypeConverter()->convertType(t);
      if (!resultType)
        return rewriter.notifyMatchFailure(op, "result type conversion failed");
      convertedResultTypes.push_back(resultType);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        op, convertedResultTypes, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

struct CustomConvertAlloc final : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp op, OpAdaptor operands,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = op.getType();

    Type eltType = memrefType.getElementType();
    if (!llvm::isa<CiphertextType, PlaintextType>(eltType)) {
      return rewriter.notifyMatchFailure(loc, "not an OpenFHE type");
    }

    Type convertedElementType = getTypeConverter()->convertType(eltType);
    if (!convertedElementType) {
      return rewriter.notifyMatchFailure(loc, "failed to convert element type");
    }

    if (!memrefType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(loc, "only static shapes supported");
    }
    int64_t numElements = memrefType.getNumElements();

    Value numElementsVal =
        emitc::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                  rewriter.getIndexAttr(numElements));

    Type targetPointerType = emitc::PointerType::get(convertedElementType);
    auto heirAllocCall = emitc::CallOpaqueOp::create(
        rewriter, loc, TypeRange{targetPointerType}, "heir_alloc",
        ValueRange{numElementsVal}, ArrayAttr{},
        ArrayAttr::get(rewriter.getContext(),
                       {TypeAttr::get(convertedElementType)}));

    rewriter.replaceOp(op, heirAllocCall.getResults());
    return success();
  }
};

struct ConvertCloneOp final
    : public OpConversionPattern<bufferization::CloneOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      bufferization::CloneOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto memrefType = cast<MemRefType>(op.getType());

    if (!memrefType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(loc,
                                         "only static shapes are supported");
    }

    Type eltType = memrefType.getElementType();
    Type convertedEltType = getTypeConverter()->convertType(eltType);
    if (!convertedEltType) return failure();

    int64_t numElements = memrefType.getNumElements();
    bool isOpenFHEType = llvm::isa<CiphertextType, PlaintextType>(eltType);

    if (isOpenFHEType && numElements > 1) {
      return rewriter.notifyMatchFailure(
          loc, "OpenFHE clone with size > 1 not supported yet");
    }

    auto ctx = rewriter.getContext();
    auto sizeTType = emitc::SizeTType::get(ctx);
    IndexType indexType = rewriter.getIndexType();

    auto numElementsVal = emitc::ConstantOp::create(
        rewriter, loc, sizeTType, rewriter.getIndexAttr(numElements));

    auto sizeofOp = emitc::CallOpaqueOp::create(
        rewriter, loc, sizeTType, "sizeof", ValueRange{},
        /*args=*/rewriter.getArrayAttr({TypeAttr::get(convertedEltType)}),
        /*template_args=*/ArrayAttr{});
    Value elementSize = sizeofOp.getResult(0);

    auto totalSizeOp = emitc::MulOp::create(rewriter, loc, sizeTType,
                                            elementSize, numElementsVal);
    Value totalSize = totalSizeOp.getResult();

    Value destPtr;
    if (isOpenFHEType) {
      Type targetPointerType = emitc::PointerType::get(convertedEltType);
      auto heirAllocCall = emitc::CallOpaqueOp::create(
          rewriter, loc, TypeRange{targetPointerType}, "heir_alloc",
          ValueRange{numElementsVal}, ArrayAttr{},
          ArrayAttr::get(rewriter.getContext(),
                         {TypeAttr::get(convertedEltType)}));
      destPtr = heirAllocCall.getResult(0);
    } else {
      auto voidType = emitc::OpaqueType::get(ctx, "void");
      auto voidPtrType = emitc::PointerType::get(voidType);

      auto mallocOp = emitc::CallOpaqueOp::create(
          rewriter, loc, voidPtrType, "malloc", ValueRange{totalSize});
      Value voidPtr = mallocOp.getResult(0);

      auto destPtrType = emitc::PointerType::get(convertedEltType);
      auto castOp = emitc::CastOp::create(rewriter, loc, destPtrType, voidPtr);
      destPtr = castOp.getResult();
    }

    Value sourcePtr = adaptor.getInput();

    if (numElements == 1) {
      auto c0Attr = rewriter.getIndexAttr(0);
      auto c0Op = emitc::ConstantOp::create(rewriter, loc, indexType, c0Attr);
      Value c0 = c0Op.getResult();

      auto typedSource = llvm::cast<TypedValue<emitc::PointerType>>(sourcePtr);
      auto typedDest = llvm::cast<TypedValue<emitc::PointerType>>(destPtr);

      auto srcSubscript =
          emitc::SubscriptOp::create(rewriter, loc, typedSource, c0);
      auto srcLoad =
          emitc::LoadOp::create(rewriter, loc, convertedEltType, srcSubscript);

      auto targetSubscript =
          emitc::SubscriptOp::create(rewriter, loc, typedDest, c0);

      emitc::AssignOp::create(rewriter, loc, targetSubscript, srcLoad);
    } else {
      emitc::CallOpaqueOp::create(rewriter, loc, TypeRange{}, "memcpy",
                                  ValueRange{destPtr, sourcePtr, totalSize});
    }

    rewriter.replaceOp(op, destPtr);
    return success();
  }
};

struct CustomConvertDealloc final
    : public OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::DeallocOp op, OpAdaptor operands,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto memrefType = cast<MemRefType>(op.getMemref().getType());

    Type eltType = memrefType.getElementType();
    if (!llvm::isa<CiphertextType, PlaintextType>(eltType)) {
      return rewriter.notifyMatchFailure(loc, "not an OpenFHE type");
    }

    Type convertedElementType = getTypeConverter()->convertType(eltType);
    if (!convertedElementType) {
      return rewriter.notifyMatchFailure(loc, "failed to convert element type");
    }

    Value memrefPtr = operands.getMemref();

    emitc::CallOpaqueOp::create(
        rewriter, loc, TypeRange{}, "heir_free", ValueRange{memrefPtr},
        ArrayAttr{},
        ArrayAttr::get(rewriter.getContext(),
                       {TypeAttr::get(convertedElementType)}));

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertReinterpretCast
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::ReinterpretCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    llvm::errs() << "DEBUG: ConvertReinterpretCast matchAndRewrite\n";
    Location loc = op.getLoc();
    auto convertedSource = adaptor.getSource();
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return failure();

    auto mixedOffsets = op.getMixedOffsets();
    llvm::errs() << "DEBUG: mixedOffsets size: " << mixedOffsets.size() << "\n";
    if (mixedOffsets.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only 1D reinterpret_cast is supported currently");
    }

    OpFoldResult ofr = mixedOffsets[0];
    Value offsetVal;
    if (ofr.is<Value>()) {
      llvm::errs() << "DEBUG: offset is Value\n";
      offsetVal = adaptor.getOffsets()[0];
      if (!offsetVal) {
        llvm::errs() << "DEBUG: offsetVal is NULL in adaptor!\n";
      } else {
        llvm::errs() << "DEBUG: offsetVal type: ";
        offsetVal.getType().print(llvm::errs());
        llvm::errs() << "\n";
      }
    } else {
      llvm::errs() << "DEBUG: offset is Attribute\n";
      auto attr = ofr.get<Attribute>();
      auto intAttr = llvm::cast<IntegerAttr>(attr);
      int64_t staticOffset = intAttr.getInt();
      llvm::errs() << "DEBUG: staticOffset: " << staticOffset << "\n";
      if (staticOffset == 0) {
        offsetVal = nullptr;
      } else {
        offsetVal =
            emitc::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                      rewriter.getIndexAttr(staticOffset));
      }
    }

    Value replaced = convertedSource;
    if (offsetVal) {
      llvm::errs() << "DEBUG: generating subscript and address_of\n";
      auto typedPtr = llvm::cast<TypedValue<emitc::PointerType>>(replaced);
      auto subscript =
          emitc::SubscriptOp::create(rewriter, loc, typedPtr, offsetVal);
      Type eltType = typedPtr.getType().getPointee();
      auto addrOp = emitc::AddressOfOp::create(
          rewriter, loc, emitc::PointerType::get(eltType), subscript);
      replaced = addrOp.getResult();
    } else {
      llvm::errs() << "DEBUG: offsetVal is null, no offset generated\n";
    }

    if (replaced.getType() != resultType) {
      replaced =
          emitc::CastOp::create(rewriter, op.getLoc(), resultType, replaced);
    }
    rewriter.replaceOp(op, replaced);
    return success();
  }
};

struct ConvertExtractAlignedPointerAsIndex
    : public OpConversionPattern<memref::ExtractAlignedPointerAsIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto convertedSource = adaptor.getSource();
    if (!llvm::isa<emitc::PointerType>(convertedSource.getType())) {
      return rewriter.notifyMatchFailure(
          op, "expected converted source to be emitc.ptr");
    }

    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return failure();

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, resultType, convertedSource);
    return success();
  }
};

struct ConvertExtractStridedMetadata
    : public OpConversionPattern<memref::ExtractStridedMetadataOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::ExtractStridedMetadataOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto memrefType = llvm::cast<MemRefType>(op.getSource().getType());

    auto convertedSource = adaptor.getSource();
    auto baseBufferType =
        getTypeConverter()->convertType(op.getBaseBuffer().getType());
    if (!baseBufferType) return failure();

    Value baseBuffer = convertedSource;
    if (baseBuffer.getType() != baseBufferType) {
      baseBuffer =
          rewriter.create<emitc::CastOp>(loc, baseBufferType, baseBuffer);
    }

    auto [strides, offset] = memrefType.getStridesAndOffset();
    if (ShapedType::isDynamic(offset)) {
      return rewriter.notifyMatchFailure(op, "dynamic offset not supported");
    }

    Value offsetVal = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(offset));

    SmallVector<Value> sizeVals;
    for (auto size : memrefType.getShape()) {
      if (ShapedType::isDynamic(size)) {
        return rewriter.notifyMatchFailure(op, "dynamic size not supported");
      }
      sizeVals.push_back(rewriter.create<emitc::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(size)));
    }

    SmallVector<Value> strideVals;
    for (auto stride : strides) {
      if (ShapedType::isDynamic(stride)) {
        return rewriter.notifyMatchFailure(op, "dynamic stride not supported");
      }
      strideVals.push_back(rewriter.create<emitc::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(stride)));
    }

    SmallVector<Value> replacements;
    replacements.push_back(baseBuffer);
    replacements.push_back(offsetVal);
    replacements.append(sizeVals.begin(), sizeVals.end());
    replacements.append(strideVals.begin(), strideVals.end());

    rewriter.replaceOp(op, replacements);
    return success();
  }
};

}  // namespace

void OpenfheToEmitCDialectInterface::populateConvertToEmitCConversionPatterns(
    mlir::ConversionTarget& target, mlir::TypeConverter& typeConverter,
    mlir::RewritePatternSet& patterns, std::optional<bool> lowerToCpp) const {
  llvm::errs() << "DEBUG: populating OpenfheToEmitC patterns\n";
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
          return emitc::OpaqueType::get(context, "PublicKeyT");
        })
        .Case<PrivateKeyType>([context](PrivateKeyType ty) {
          return emitc::OpaqueType::get(context, "PrivateKeyT");
        })
        .Case<DigitDecompositionType>([context](DigitDecompositionType ty) {
          return emitc::OpaqueType::get(context, "FastRotPrecompT");
        })
        .Case<RankedTensorType>([&typeConverter](RankedTensorType ty) -> Type {
          auto eltType = ty.getElementType();
          auto convertedEltType = typeConverter.convertType(eltType);
          if (!convertedEltType) return nullptr;
          return emitc::PointerType::get(convertedEltType);
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
  patterns.add<MemberCallConversion<AddPlainOp>>(
      typeConverter, context, "EvalAdd", returnsResultResolver);
  patterns.add<MemberCallConversion<SubPlainOp>>(
      typeConverter, context, "EvalSub", returnsResultResolver);
  patterns.add<MemberCallConversion<ModReduceOp>>(
      typeConverter, context, "ModReduce", returnsResultResolver);
  patterns.add<MemberCallConversion<BootstrapOp>>(
      typeConverter, context, "EvalBootstrap", returnsResultResolver);

  patterns.add<MemberCallConversion<AddInPlaceOp>>(
      typeConverter, context, "EvalAddInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<SubInPlaceOp>>(
      typeConverter, context, "EvalSubInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<AddPlainInPlaceOp>>(
      typeConverter, context, "EvalAddInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<SubPlainInPlaceOp>>(
      typeConverter, context, "EvalSubInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<RelinInPlaceOp>>(
      typeConverter, context, "RelinearizeInPlace", inPlaceResolver,
      /*isVoidCall=*/true);
  patterns.add<MemberCallConversion<ModReduceInPlaceOp>>(
      typeConverter, context, "ModReduceInPlace", inPlaceResolver,
      /*isVoidCall=*/true);

  patterns.add<
      ConvertGenBootstrapKeyOp, ConvertSetupBootstrapOp, ConvertGenParamsOp,
      ConvertGenContextOp, ConvertEncryptOp, ConvertGenMulKeyOp,
      ConvertMakePackedPlaintextOp, ConvertMakeCKKSPackedPlaintextOp,
      ConvertRotOp, ConvertDecryptOp, ConvertGenRotKeyOp, ConvertDecodeOp,
      ConvertDecodeCKKSOp, ConvertFastRotationOp,
      ConvertFastRotationPrecomputeOp, ConvertCopy, ConvertGlobal,
      ConvertGetGlobal, CustomCallOpConversion, ConvertToTensorOp,
      ConvertToBufferOp, CustomLoadOpConversion, CustomStoreOpConversion,
      ConvertFloorDivSIOp, ConvertMemRefCast, ConvertMemRefExpandShape,
      ConvertMemRefCollapseShape, ConvertMemRefSubView, ConvertExtFOp,
      ConvertModule, ConvertLevelReduceOp, ConvertLevelReduceInPlaceOp>(
      typeConverter, context);
  patterns.add<CustomConvertAlloc>(typeConverter, context,
                                   mlir::PatternBenefit(2));
  patterns.add<CustomConvertDealloc>(typeConverter, context,
                                     mlir::PatternBenefit(2));
  patterns.add<ConvertCloneOp>(typeConverter, context);
  patterns.add<ConvertExtractAlignedPointerAsIndex,
               ConvertExtractStridedMetadata, ConvertReinterpretCast>(
      typeConverter, context);

  target.addIllegalDialect<OpenfheDialect>();
}

}  // namespace mlir::heir::openfhe
