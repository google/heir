#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETTOMODARITH
#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h.inc"

bool isModArithOrContainerOfModArith(Type type) {
  return isa<mod_arith::ModArithType>(getElementTypeOrSelf(type));
}

class SecretToModArithTypeConverter : public TypeConverter {
 public:
  SecretToModArithTypeConverter(MLIRContext *ctx, int64_t ptm)
      : plaintextModulus(ptm) {
    addConversion([](Type type) { return type; });
    addConversion(
        [this](secret::SecretType type) { return convertSecretType(type); });
  }

  Type convertPlaintextType(Type type) const {
    DenseMap<int64_t, int64_t> significantBitSize = {
        {64, 53},  // f64
        {32, 24},  // f32
        {16, 11}   // f16
    };

    auto *ctx = type.getContext();
    return TypeSwitch<Type, Type>(type)
        .Case<ShapedType>([this](ShapedType shapedType) {
          return shapedType.cloneWith(
              shapedType.getShape(),
              convertPlaintextType(shapedType.getElementType()));
        })
        .Case<IntegerType>([this, ctx](IntegerType intType) {
          Type newType;
          int64_t mod = plaintextModulus;
          if (plaintextModulus == 0) {
            auto modulusBitSize = (int64_t)intType.getIntOrFloatBitWidth();
            mod = (1L << (modulusBitSize - 1L));
            newType = mlir::IntegerType::get(intType.getContext(),
                                             modulusBitSize + 1);
          } else {
            newType = mlir::IntegerType::get(ctx, 64);
          }

          return mod_arith::ModArithType::get(
              ctx, mlir::IntegerAttr::get(newType, mod));
        })
        // For the float types below, using as the modulus the natural bit size
        // of the significant (including the implicit sign bit).
        .Case<FloatType>([&](FloatType floatType) {
          Type newType = mlir::IntegerType::get(ctx, 64);
          int64_t modulus = 1L << significantBitSize[floatType.getWidth()];
          return mod_arith::ModArithType::get(
              ctx, mlir::IntegerAttr::get(newType, modulus));
        })
        .Default([](Type t) { return t; });
  }

  Type convertSecretType(secret::SecretType type) const {
    return convertPlaintextType(type.getValueType());
  }

 private:
  int64_t plaintextModulus;
};

template <typename T, typename Y = T>
class SecretGenericOpConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  SecretGenericOpConversion(const TypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<secret::GenericOp>(typeConverter, context,
                                               benefit) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (op.getBody()->getOperations().size() > 2) {
      // Each secret.generic should contain at most one instruction -
      // secret-distribute-generic can be used to distribute through the
      // arithmetic ops.
      return failure();
    }

    auto &innerOp = op.getBody()->getOperations().front();
    if (!isa<T>(innerOp)) {
      return failure();
    }

    // The inner op's arguments are either plaintext operands, in which case
    // they are already type-converted, or else they are ciphertext operands,
    // in which case we can get them in type-converted form from the adaptor.
    SmallVector<Value> inputs;
    for (Value operand : innerOp.getOperands()) {
      if (auto *secretArg = op.getOpOperandForBlockArgument(operand)) {
        inputs.push_back(adaptor.getInputs()[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand);
      }
    }

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    FailureOr<Operation *> newOpResult =
        matchAndRewriteInner(op, resultTypes, inputs, rewriter);
    if (failed(newOpResult)) return failure();
    return success();
  }

  // Default method for replacing the secret.generic with the target
  // operation.
  virtual FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter &rewriter) const {
    return rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs)
        .getOperation();
  }
};

// This is similar to ConversionUtils::convertAnyOperand, but it requires the
// cloning to occur on the op inside the secret generic, while using
// type-converted operands and results of the outer generic op.
class ConvertAnyNestedGeneric : public OpConversionPattern<secret::GenericOp> {
 public:
  ConvertAnyNestedGeneric(const TypeConverter &typeConverter,
                          MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<secret::GenericOp>(typeConverter, context,
                                               benefit) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp outerOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (outerOp.getBody()->getOperations().size() > 2) {
      return failure();
    }
    Operation *innerOp = &outerOp.getBody()->getOperations().front();

    SmallVector<Value> inputs;
    for (Value operand : innerOp->getOperands()) {
      if (auto *secretArg = outerOp.getOpOperandForBlockArgument(operand)) {
        inputs.push_back(adaptor.getInputs()[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand);
      }
    }

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(outerOp.getResultTypes(),
                                                resultTypes)))
      return failure();

    SmallVector<std::unique_ptr<Region>, 1> regions;
    IRMapping mapping;
    for (auto &r : innerOp->getRegions()) {
      Region *newRegion = new Region(innerOp);
      rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
      if (failed(rewriter.convertRegionTypes(newRegion, *typeConverter)))
        return failure();
      regions.emplace_back(newRegion);
    }

    Operation *newOp = rewriter.create(OperationState(
        outerOp.getLoc(), innerOp->getName().getStringRef(), inputs,
        resultTypes, innerOp->getAttrs(), innerOp->getSuccessors(), regions));
    rewriter.replaceOp(outerOp, newOp);
    return success();
  }
};

LogicalResult ensureFloat(Value value) {
  Type cleartextEltTy = getElementTypeOrSelf(value.getType());
  return isa<FloatType>(cleartextEltTy) ? success() : failure();
}

// Construct a FloatAttr corresponding to 2^logScale, of the correct shaped
// type matching the input value.
TypedAttr getScaleAttr(Value value, int64_t logScale, ImplicitLocOpBuilder &b) {
  auto floatEltTy = cast<FloatType>(getElementTypeOrSelf(value.getType()));
  APFloat scale(floatEltTy.getFloatSemantics(), 1 << logScale);
  if (auto shapedTy = dyn_cast<ShapedType>(value.getType())) {
    return DenseFPElementsAttr::get(shapedTy, scale);
  }

  return b.getFloatAttr(floatEltTy, scale);
}

// "encode" a cleartext to mod_arith by sign extending and encapsulating it.
Value encodeCleartext(Value cleartext, Type resultType, int64_t logScale,
                      ImplicitLocOpBuilder &b) {
  // We start with something like an i16 (or tensor<Nxi16>) and the result
  // should be a (tensor of) !mod_arith.int<17 : i64> so we need to first
  // sign extend the input to the mod_arith storage type, then encapsulate it
  // into the mod_arith type.
  mod_arith::ModArithType resultEltTy =
      cast<mod_arith::ModArithType>(getElementTypeOrSelf(resultType));
  IntegerType modulusType =
      cast<IntegerType>(resultEltTy.getModulus().getType());
  Type extendedType = modulusType;

  if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
    extendedType = shapedType.cloneWith(shapedType.getShape(), modulusType);
  }

  if (isa<FloatType>(getElementTypeOrSelf(cleartext.getType()))) {
    auto scaleOp = b.create<arith::MulFOp>(
        cleartext,
        b.create<arith::ConstantOp>(getScaleAttr(cleartext, logScale, b)));
    auto convertToIntOp = b.create<arith::FPToSIOp>(extendedType, scaleOp);
    auto encapsulateOp = b.create<mod_arith::EncapsulateOp>(
        resultType, convertToIntOp.getResult());
    return encapsulateOp.getResult();
  }

  auto extOp = b.create<arith::ExtSIOp>(extendedType, cleartext);
  auto encapsulateOp =
      b.create<mod_arith::EncapsulateOp>(resultType, extOp.getResult());
  return encapsulateOp.getResult();
}

struct ConvertConceal : public OpConversionPattern<secret::ConcealOp> {
  ConvertConceal(const SecretToModArithTypeConverter &typeConverter,
                 mlir::MLIRContext *context, PatternBenefit benefit,
                 int64_t logScale)
      : OpConversionPattern<secret::ConcealOp>(context, benefit),
        logScale(logScale),
        typeConv(typeConverter) {}

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, secret::ConcealOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // We start with something like an i16 (or tensor<Nxi16>) and the result
    // should be a (tensor of) !mod_arith.int<17 : i64> so we need to first
    // sign extend the input to the mod_arith storage type, then encapsulate it
    // into the mod_arith type.
    if (logScale && failed(ensureFloat(adaptor.getCleartext()))) {
      return op->emitError() << "Conceal op requires a floating point "
                                "cleartext when logScale is set";
    }
    Type convertedType = typeConv.convertSecretType(op.getResult().getType());
    Value replacementValue =
        encodeCleartext(adaptor.getCleartext(), convertedType, logScale, b);
    rewriter.replaceOp(op, replacementValue);
    return success();
  }

 private:
  int64_t logScale;
  const SecretToModArithTypeConverter &typeConv;
};

struct ConvertReveal : public OpConversionPattern<secret::RevealOp> {
  ConvertReveal(const SecretToModArithTypeConverter &typeConverter,
                mlir::MLIRContext *context, PatternBenefit benefit,
                int64_t logScale)
      : OpConversionPattern<secret::RevealOp>(typeConverter, context, benefit),
        logScale(logScale) {}

  LogicalResult matchAndRewrite(
      secret::RevealOp op, secret::RevealOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    if (logScale && failed(ensureFloat(op.getResult()))) {
      return op->emitError() << "Reveal op requires a floating point "
                                "cleartext when logScale is set";
    }

    // We start with something like a secret<i16> (or secret<tensor<Nxi16>>)
    // and type conversion gives us a (tensor of) !mod_arith.int<17 : i64> so
    // we need to first mod_arith extract to get the result in terms of i64,
    // then truncate to the original i16 type.
    Type modArithTypeOrTensor = adaptor.getInput().getType();
    auto eltTy = cast<mod_arith::ModArithType>(
        getElementTypeOrSelf(modArithTypeOrTensor));
    IntegerType modulusType = cast<IntegerType>(eltTy.getModulus().getType());
    Type beforeTrunc = modulusType;
    if (auto shapedType = dyn_cast<ShapedType>(modArithTypeOrTensor)) {
      beforeTrunc = shapedType.cloneWith(shapedType.getShape(), modulusType);
    }
    Type truncatedType = op.getResult().getType();

    auto extractOp =
        b.create<mod_arith::ExtractOp>(beforeTrunc, adaptor.getInput());
    Value result = extractOp.getResult();

    Type cleartextEltTy = getElementTypeOrSelf(op.getResult().getType());
    if (isa<FloatType>(cleartextEltTy)) {
      // convert to float then undo scale.
      auto convertOp =
          b.create<arith::SIToFPOp>(op.getResult().getType(), result);
      auto scaleValueOp = b.create<arith::ConstantOp>(
          op.getLoc(), getScaleAttr(op.getResult(), logScale, b));
      auto scaleOp = b.create<arith::DivFOp>(convertOp, scaleValueOp);
      result = scaleOp.getResult();
    } else {
      auto truncOp = b.create<arith::TruncIOp>(truncatedType, result);
      result = truncOp.getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }

 private:
  int64_t logScale;
};

// This is like
// ContextAwareConversionUtils::SecretGenericOpCipherPlainConversion, except
// that secret types are type converted to mod_arith, while plaintext types
// stay as regular tensor types, and need to be "encoded" (encapsulated) into
// mod_arith tensors, whereas for normal secret-to-scheme, there is a dedicated
// ciphertext-plaintext op.
template <typename T, typename Y>
class SecretGenericOpCipherPlainConversion
    : public SecretGenericOpConversion<T, Y> {
 public:
  // Ciphertext-plaintext ops should take precedence over ciphertext-ciphertext
  // ops because the ops being converted (e.g., addi) don't have a plaintext
  // variant.
  SecretGenericOpCipherPlainConversion(const TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       PatternBenefit benefit, int64_t logScale)
      : SecretGenericOpConversion<T, Y>(typeConverter, context, benefit),
        logScale(logScale) {}

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter &rewriter) const override {
    // Verify that exactly one of the two inputs is a ciphertext.
    if (inputs.size() != 2 ||
        llvm::count_if(inputs, [&](Value input) {
          return isModArithOrContainerOfModArith(input.getType());
        }) != 1) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    if (isModArithOrContainerOfModArith(input0.getType())) {
      if (logScale && failed(ensureFloat(input1))) {
        return op->emitError() << "Cleartext value requires a floating point "
                                  "type when logScale is set";
      }
      auto encoded = encodeCleartext(input1, input0.getType(), logScale, b);
      auto newOp = rewriter.replaceOpWithNewOp<Y>(op, input0, encoded);
      return newOp.getOperation();
    }

    if (logScale && failed(ensureFloat(input0))) {
      return op->emitError() << "Cleartext value requires a floating point "
                                "type when logScale is set";
    }
    auto encoded = encodeCleartext(input0, input1.getType(), logScale, b);
    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, encoded, input1);
    return newOp.getOperation();
  }

 private:
  int64_t logScale;
};

// For floating point cleartexts, the plaintext backend requires converting
// float types to mod_arith types, which hence requires a scale parameter
// to discretize the floats. This in turn requires scale management/tracking
// in the IR, and to avoid overflow we reuse the scale analysis/mod_reduce
// ops to restore the original scale after each mul.
//
// mod_reduce just lowers to a division operation.
struct ConvertModReduce : public SecretGenericOpConversion<mgmt::ModReduceOp> {
  ConvertModReduce(const SecretToModArithTypeConverter &typeConverter,
                   mlir::MLIRContext *context, PatternBenefit benefit,
                   int64_t logScale)
      : SecretGenericOpConversion<mgmt::ModReduceOp>(typeConverter, context,
                                                     benefit),
        logScale(logScale) {}

  FailureOr<Operation *> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto innerOp =
        cast<mgmt::ModReduceOp>(op.getBody()->getOperations().front());
    if (logScale && failed(ensureFloat(innerOp.getInput()))) {
      return op->emitError() << "requires a floating point "
                                "input when logScale is set";
    }

    // We make the assumption that the scale analysis earlier in the pipeline
    // will always mod_reduce after each mul, so the scale is always being
    // reduced from 2^(2*logScale) to 2^logScale. This means mod_reduce is
    // always a division by (2^log_scale).
    mod_arith::ModArithType modArithType = cast<mod_arith::ModArithType>(
        getElementTypeOrSelf(inputs[0].getType()));
    APInt modulus = modArithType.getModulus().getValue();
    APInt scale = APInt(modulus.getBitWidth(), 1) << logScale;

    if (scale.isNonPositive()) {
      return op->emitError()
             << "scale must be positive, got " << scale.getSExtValue()
             << " for modulus " << modulus.getSExtValue() << " and logScale "
             << logScale;
    }

    APInt inverseScale = multiplicativeInverse(scale, modulus);

    Type inverseScaleTy = modArithType.getModulus().getType();
    TypedAttr inverseScaleAttr = IntegerAttr::get(inverseScaleTy, inverseScale);
    if (auto shapedType =
            dyn_cast<ShapedType>(modArithType.getModulus().getType())) {
      inverseScaleTy = shapedType.cloneWith(
          shapedType.getShape(), modArithType.getModulus().getType());
      inverseScaleAttr = DenseElementsAttr::get(shapedType, inverseScaleAttr);
    }
    auto scaleValueOp =
        b.create<arith::ConstantOp>(inverseScaleTy, inverseScaleAttr);
    auto encapsulateOp = b.create<mod_arith::EncapsulateOp>(
        outputTypes[0], scaleValueOp.getResult());
    auto divOp = rewriter.replaceOpWithNewOp<mod_arith::MulOp>(
        op, inputs[0], encapsulateOp.getResult());
    return divOp.getOperation();
  }

 private:
  int64_t logScale;
};

// Similar to ConvertModReduce, we need to encode static cleartexts mid-IR
// so they have the same scale as other plaintext values mid computation.
// The scale analysis provides this by inserting mgmt.init operations, so
// in this case mgmt.init lowers to a mul op by the scale.
struct ConvertInit : public OpConversionPattern<mgmt::InitOp> {
  ConvertInit(const SecretToModArithTypeConverter &typeConverter,
              mlir::MLIRContext *context, PatternBenefit benefit,
              int64_t logScale)
      : OpConversionPattern<mgmt::InitOp>(typeConverter, context, benefit),
        logScale(logScale) {}

  LogicalResult matchAndRewrite(
      mgmt::InitOp op, mgmt::InitOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    if (logScale && failed(ensureFloat(op.getInput()))) {
      return op->emitError() << "requires a floating point "
                                "input when logScale is set";
    }
    auto scaleValueOp = b.create<arith::ConstantOp>(
        op.getLoc(), getScaleAttr(op.getResult(), logScale, b));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, adaptor.getInput(),
                                               scaleValueOp.getResult());
    return success();
  }

 private:
  int64_t logScale;
};

struct SecretToModArith : public impl::SecretToModArithBase<SecretToModArith> {
  using SecretToModArithBase::SecretToModArithBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    SecretToModArithTypeConverter typeConverter(context, plaintextModulus);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<mod_arith::ModArithDialect>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalDialect<mgmt::MgmtDialect>();

    // These patterns have higher benefit to take precedence over the default
    // pattern, which simply converts operand/result types and inlines the
    // operation inside the generic.
    patterns.add<
        SecretGenericOpCipherPlainConversion<arith::AddFOp, mod_arith::AddOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, mod_arith::AddOp>,
        SecretGenericOpCipherPlainConversion<arith::MulFOp, mod_arith::MulOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, mod_arith::MulOp>,
        SecretGenericOpCipherPlainConversion<arith::SubFOp, mod_arith::SubOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, mod_arith::SubOp>,
        ConvertReveal, ConvertConceal, ConvertModReduce, ConvertInit>(
        typeConverter, context, /*benefit=*/3, logScale);

    patterns.add<SecretGenericOpConversion<arith::AddIOp, mod_arith::AddOp>,
                 SecretGenericOpConversion<arith::SubIOp, mod_arith::SubOp>,
                 SecretGenericOpConversion<arith::MulIOp, mod_arith::MulOp>,
                 SecretGenericOpConversion<arith::AddFOp, mod_arith::AddOp>,
                 SecretGenericOpConversion<arith::SubFOp, mod_arith::SubOp>,
                 SecretGenericOpConversion<arith::MulFOp, mod_arith::MulOp>>(
        typeConverter, context,
        /*benefit=*/2);

    patterns.add<ConvertAnyNestedGeneric>(typeConverter, context,
                                          /*benefit=*/1);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clear any tensor_ext attributes from the func
    getOperation()->walk([&](FunctionOpInterface funcOp) {
      for (int i = 0; i < funcOp.getNumArguments(); ++i) {
        for (auto attr : funcOp.getArgAttrs(i)) {
          // the attr name is tensor_ext.foo, so just check for the prefix
          if (attr.getName().getValue().starts_with("tensor_ext")) {
            funcOp.removeArgAttr(i, attr.getName());
          }
        }
      }

      for (int i = 0; i < funcOp.getNumResults(); ++i) {
        for (auto attr : funcOp.getResultAttrs(i)) {
          if (attr.getName().getValue().starts_with("tensor_ext")) {
            funcOp.removeResultAttr(i, attr.getName());
          }
        }
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
