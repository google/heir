#include "lib/Dialect/Secret/Conversions/SecretToModArith/SecretToModArith.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
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
  SecretToModArithTypeConverter(MLIRContext* ctx, int64_t ptm)
      : plaintextModulus(ptm) {
    addConversion([](Type type) { return type; });
    addConversion(
        [this](secret::SecretType type) { return convertSecretType(type); });
  }

  Type convertPlaintextType(Type type) const {
    auto* ctx = type.getContext();
    return TypeSwitch<Type, Type>(type)
        .Case<ShapedType>([this](ShapedType shapedType) {
          return shapedType.cloneWith(
              shapedType.getShape(),
              convertPlaintextType(shapedType.getElementType()));
        })
        .Case<IntegerType>([this, ctx](IntegerType intType) -> Type {
          if (plaintextModulus == 0) {
            return intType;
          }

          int64_t mod = plaintextModulus;
          Type newType = mlir::IntegerType::get(ctx, 64);
          return mod_arith::ModArithType::get(
              ctx, mlir::IntegerAttr::get(newType, mod));
        })
        // Default includes FloatType, which is allowed and unchanged for debug
        // purposes.
        .Default([](Type t) { return t; });
  }

  Type convertSecretType(secret::SecretType type) const {
    return convertPlaintextType(type.getValueType());
  }

  int64_t getPlaintextModulus() const { return plaintextModulus; }

 private:
  int64_t plaintextModulus;
};

template <typename T, typename Y = T>
class SecretGenericOpConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  SecretGenericOpConversion(const TypeConverter& typeConverter,
                            MLIRContext* context, PatternBenefit benefit = 1)
      : OpConversionPattern<secret::GenericOp>(typeConverter, context,
                                               benefit) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (op.getBody()->getOperations().size() > 2) {
      // Each secret.generic should contain at most one instruction -
      // secret-distribute-generic can be used to distribute through the
      // arithmetic ops.
      return failure();
    }

    auto& innerOp = op.getBody()->getOperations().front();
    if (!isa<T>(innerOp)) {
      return failure();
    }

    // The inner op's arguments are either plaintext operands, in which case
    // they are already type-converted, or else they are ciphertext operands,
    // in which case we can get them in type-converted form from the adaptor.
    SmallVector<Value> inputs;
    for (Value operand : innerOp.getOperands()) {
      if (auto* secretArg = op.getOpOperandForBlockArgument(operand)) {
        inputs.push_back(adaptor.getInputs()[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand);
      }
    }

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    FailureOr<Operation*> newOpResult =
        matchAndRewriteInner(op, resultTypes, inputs, rewriter);
    if (failed(newOpResult)) return failure();
    return success();
  }

  // Default method for replacing the secret.generic with the target
  // operation.
  virtual FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter& rewriter) const {
    return rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs)
        .getOperation();
  }
};

// This is similar to ConversionUtils::convertAnyOperand, but it requires the
// cloning to occur on the op inside the secret generic, while using
// type-converted operands and results of the outer generic op.
class ConvertAnyNestedGeneric : public OpConversionPattern<secret::GenericOp> {
 public:
  ConvertAnyNestedGeneric(const TypeConverter& typeConverter,
                          MLIRContext* context, PatternBenefit benefit = 1)
      : OpConversionPattern<secret::GenericOp>(typeConverter, context,
                                               benefit) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp outerOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (outerOp.getBody()->getOperations().size() > 2) {
      return failure();
    }
    Operation* innerOp = &outerOp.getBody()->getOperations().front();

    SmallVector<Value> inputs;
    for (Value operand : innerOp->getOperands()) {
      if (auto* secretArg = outerOp.getOpOperandForBlockArgument(operand)) {
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
    for (auto& r : innerOp->getRegions()) {
      Region* newRegion = new Region(innerOp);
      rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
      if (failed(rewriter.convertRegionTypes(newRegion, *typeConverter)))
        return failure();
      regions.emplace_back(newRegion);
    }

    Operation* newOp = rewriter.create(OperationState(
        outerOp.getLoc(), innerOp->getName().getStringRef(), inputs,
        resultTypes, innerOp->getAttrs(), innerOp->getSuccessors(), regions));
    rewriter.replaceOp(outerOp, newOp);
    return success();
  }
};

// "encode" a cleartext to mod_arith by sign extending and encapsulating it.
// If the cleartext is a float type, let it pass through unchanged.
static Value encodeCleartext(Value cleartext, Type resultType,
                             ImplicitLocOpBuilder& b) {
  if (isa<FloatType>(getElementTypeOrSelf(cleartext.getType()))) {
    return cleartext;
  }
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

  auto extOp = arith::ExtSIOp::create(b, extendedType, cleartext);
  auto encapsulateOp =
      mod_arith::EncapsulateOp::create(b, resultType, extOp.getResult());
  return encapsulateOp.getResult();
}

struct ConvertConceal : public OpConversionPattern<secret::ConcealOp> {
  ConvertConceal(const SecretToModArithTypeConverter& typeConverter,
                 mlir::MLIRContext* context, PatternBenefit benefit)
      : OpConversionPattern<secret::ConcealOp>(context, benefit),
        typeConv(typeConverter) {}

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, secret::ConcealOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    if (typeConv.getPlaintextModulus() == 0) {
      // If the plaintext modulus is 0, we can just pass through the
      // cleartext without any modifications.
      rewriter.replaceOp(op, adaptor.getCleartext());
      return success();
    }
    // We start with something like an i16 (or tensor<Nxi16>) and the result
    // should be a (tensor of) !mod_arith.int<17 : i64> so we need to first
    // sign extend the input to the mod_arith storage type, then encapsulate it
    // into the mod_arith type.
    Type convertedType = typeConv.convertSecretType(op.getResult().getType());
    Value replacementValue =
        encodeCleartext(adaptor.getCleartext(), convertedType, b);
    rewriter.replaceOp(op, replacementValue);
    return success();
  }

 private:
  const SecretToModArithTypeConverter& typeConv;
};

struct ConvertReveal : public OpConversionPattern<secret::RevealOp> {
  ConvertReveal(const SecretToModArithTypeConverter& typeConverter,
                mlir::MLIRContext* context, PatternBenefit benefit)
      : OpConversionPattern<secret::RevealOp>(typeConverter, context, benefit),
        typeConv(typeConverter) {}

  LogicalResult matchAndRewrite(
      secret::RevealOp op, secret::RevealOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (isa<FloatType>(getElementTypeOrSelf(adaptor.getInput().getType()))) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }
    if (typeConv.getPlaintextModulus() == 0) {
      // If the plaintext modulus is 0, we can just pass through the
      // mod_arith.int without any modifications.
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
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

    auto extractOp =
        mod_arith::ExtractOp::create(b, beforeTrunc, adaptor.getInput());
    Value result = extractOp.getResult();

    Type truncatedType = op.getResult().getType();
    if (getElementTypeOrSelf(truncatedType).getIntOrFloatBitWidth() <
        getElementTypeOrSelf(extractOp.getResult().getType())
            .getIntOrFloatBitWidth()) {
      auto truncOp = arith::TruncIOp::create(b, truncatedType, result);
      result = truncOp.getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }

 private:
  const SecretToModArithTypeConverter& typeConv;
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
  SecretGenericOpCipherPlainConversion(const TypeConverter& typeConverter,
                                       MLIRContext* context,
                                       PatternBenefit benefit)
      : SecretGenericOpConversion<T, Y>(typeConverter, context, benefit) {}

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter& rewriter) const override {
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
      auto encoded = encodeCleartext(input1, input0.getType(), b);
      auto newOp = rewriter.replaceOpWithNewOp<Y>(op, input0, encoded);
      return newOp.getOperation();
    }

    auto encoded = encodeCleartext(input0, input1.getType(), b);
    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, encoded, input1);
    return newOp.getOperation();
  }
};

// The debug port added in secret::AddDebugPort works on the cleartext type, so
// to avoid type conflicts with the encoded type, we need to insert a
// secret.reveal before the call.
struct ConvertDebugCall : public SecretGenericOpConversion<func::CallOp> {
  using SecretGenericOpConversion<func::CallOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ConversionPatternRewriter& rewriter) const override {
    auto innerOp = cast<func::CallOp>(op.getBody()->getOperations().front());
    if (innerOp.getArgOperands().size() != 1) {
      // Debug calls have a single argument.
      return failure();
    }

    // It's a bit strange: here we're ignoring the type converted operands
    // because we want to do a reveal and let the reveal pattern handle the
    // type conversion.
    auto revealOp =
        secret::RevealOp::create(rewriter, op.getLoc(), op.getOperands()[0]);
    auto newCallOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, innerOp.getResultTypes(), innerOp.getCallee(),
        revealOp.getResult());
    return newCallOp.getOperation();
  }
};

struct SecretToModArith : public impl::SecretToModArithBase<SecretToModArith> {
  using SecretToModArithBase::SecretToModArithBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    SecretToModArithTypeConverter typeConverter(context, plaintextModulus);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<mod_arith::ModArithDialect>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalDialect<mgmt::MgmtDialect>();

    // These patterns have higher benefit to take precedence over the default
    // pattern, which simply converts operand/result types and inlines the
    // operation inside the generic.
    if (plaintextModulus != 0) {
      patterns.add<
          SecretGenericOpCipherPlainConversion<arith::AddIOp, mod_arith::AddOp>,
          SecretGenericOpCipherPlainConversion<arith::MulIOp, mod_arith::MulOp>,
          SecretGenericOpCipherPlainConversion<arith::SubIOp, mod_arith::SubOp>,
          ConvertReveal, ConvertConceal, ConvertDebugCall>(typeConverter,
                                                           context,
                                                           /*benefit=*/3);

      patterns.add<SecretGenericOpConversion<arith::AddIOp, mod_arith::AddOp>,
                   SecretGenericOpConversion<arith::SubIOp, mod_arith::SubOp>,
                   SecretGenericOpConversion<arith::MulIOp, mod_arith::MulOp>>(
          typeConverter, context,
          /*benefit=*/2);
    } else {
      patterns.add<ConvertReveal, ConvertConceal, ConvertDebugCall>(
          typeConverter, context, /*benefit=*/3);
    }

    patterns.add<ConvertAnyNestedGeneric>(typeConverter, context,
                                          /*benefit=*/1);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
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
