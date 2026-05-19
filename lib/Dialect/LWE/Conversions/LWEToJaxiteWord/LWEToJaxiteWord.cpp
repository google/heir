#include "lib/Dialect/LWE/Conversions/LWEToJaxiteWord/LWEToJaxiteWord.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

class JaxiteWordTypeConverter : public TypeConverter {
 public:
  JaxiteWordTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
      return jaxiteword::PublicKeyType::get(ctx);
    });
    addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
      return jaxiteword::PrivateKeyType::get(ctx);
    });
    addConversion([](RankedTensorType type) -> Type {
      return RankedTensorType::get(type.getShape(), type.getElementType());
    });
  }
};

#define GEN_PASS_DEF_LWETOJAXITEWORD
#include "lib/Dialect/LWE/Conversions/LWEToJaxiteWord/LWEToJaxiteWord.h.inc"

namespace {

FailureOr<Value> getContextualCryptoContextForJaxiteWord(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) return failure();
  if (funcOp.getNumArguments() == 0 ||
      !mlir::isa<jaxiteword::CryptoContextType>(
          funcOp.getArgument(0).getType())) {
    return failure();
  }
  return funcOp.getArgument(0);
}

FailureOr<Value> getContextualEvalKeyForJaxiteWord(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) return failure();
  // Based on AddCryptoContextAndKeys, EvalKey is the 2nd argument (index 1)
  if (funcOp.getNumArguments() < 2 ||
      !mlir::isa<jaxiteword::EvalKeyType>(funcOp.getArgument(1).getType())) {
    return failure();
  }
  return funcOp.getArgument(1);
}

struct AddCryptoContextAndKeys : public OpConversionPattern<func::FuncOp> {
  AddCryptoContextAndKeys(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto containsCryptoOps =
        ::mlir::heir::containsDialects<lwe::LWEDialect, ckks::CKKSDialect,
                                       bgv::BGVDialect>(op);
    if (!containsCryptoOps) return failure();

    auto cryptoContextType = jaxiteword::CryptoContextType::get(getContext());
    auto evalKeyType = jaxiteword::EvalKeyType::get(getContext());

    rewriter.modifyOpInPlace(op, [&] {
      bool hasContext = op.getFunctionType().getNumInputs() > 0 &&
                        mlir::isa<jaxiteword::CryptoContextType>(
                            op.getFunctionType().getInput(0));
      if (!hasContext) {
        if (failed(op.insertArgument(0, evalKeyType, nullptr, op.getLoc())))
          return failure();
        if (failed(
                op.insertArgument(0, cryptoContextType, nullptr, op.getLoc())))
          return failure();
      }
    });

    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct ConvertBinOp : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    rewriter.replaceOpWithNewOp<TargetOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertMulOp : public OpConversionPattern<ckks::MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    FailureOr<Value> evalKey = getContextualEvalKeyForJaxiteWord(op);
    if (failed(evalKey)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::MulOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getLhs(), adaptor.getRhs(), evalKey.value());
    return success();
  }
};

template <typename SourceOp>
struct ConvertNegateOp : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::NegateOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

struct ConvertRotateOp : public OpConversionPattern<ckks::RotateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    FailureOr<Value> evalKey = getContextualEvalKeyForJaxiteWord(op);
    if (failed(evalKey)) return failure();

    Value dynamicShift = adaptor.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift) {
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");
    }
    if (dynamicShift) {
      return rewriter.notifyMatchFailure(
          op, "jaxiteword rotation requires static shift");
    }

    rewriter.replaceOpWithNewOp<jaxiteword::RotOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), evalKey.value(), staticShift);
    return success();
  }
};

struct ConvertRelinOp : public OpConversionPattern<ckks::RelinearizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    FailureOr<Value> evalKey = getContextualEvalKeyForJaxiteWord(op);
    if (failed(evalKey)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::RelinOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), evalKey.value());
    return success();
  }
};

template <typename SourceOp>
struct ConvertModSwitchOp : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::ModReduceOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

struct ConvertEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::EncodeOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

struct ConvertEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::EncryptOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), adaptor.getKey());
    return success();
  }
};

struct ConvertDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::DecryptOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), adaptor.getSecretKey());
    return success();
  }
};

struct ConvertDecodeOp : public OpConversionPattern<lwe::RLWEDecodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEDecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> ctx = getContextualCryptoContextForJaxiteWord(op);
    if (failed(ctx)) return failure();

    rewriter.replaceOpWithNewOp<jaxiteword::DecodeOp>(
        op, this->getTypeConverter()->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

}  // namespace

struct LWEToJaxiteWord : public impl::LWEToJaxiteWordBase<LWEToJaxiteWord> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalDialect<jaxiteword::JaxiteWordDialect,
                           tensor::TensorDialect>();
    target.addIllegalDialect<lwe::LWEDialect, ckks::CKKSDialect,
                             bgv::BGVDialect>();
    target.addLegalOp<func::ReturnOp>();

    JaxiteWordTypeConverter typeConverter(context);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto containsCryptoOps =
          ::mlir::heir::containsDialects<lwe::LWEDialect, ckks::CKKSDialect,
                                         bgv::BGVDialect>(op);
      if (!containsCryptoOps) return true;
      bool hasArgs = op.getFunctionType().getNumInputs() >= 2;
      return typeConverter.isSignatureLegal(op.getFunctionType()) && hasArgs &&
             mlir::isa<jaxiteword::CryptoContextType>(
                 op.getFunctionType().getInput(0)) &&
             mlir::isa<jaxiteword::EvalKeyType>(
                 op.getFunctionType().getInput(1));
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    addTensorConversionPatterns(typeConverter, patterns, target);

    patterns.add<AddCryptoContextAndKeys>(typeConverter, context);
    patterns.add<ConvertBinOp<lwe::AddOp, jaxiteword::AddOp>>(typeConverter,
                                                              context);
    patterns.add<ConvertBinOp<lwe::RAddOp, jaxiteword::AddOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertBinOp<lwe::RSubOp, jaxiteword::SubOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertBinOp<ckks::AddOp, jaxiteword::AddOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertBinOp<ckks::SubOp, jaxiteword::SubOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMulOp>(typeConverter, context);
    patterns.add<ConvertBinOp<lwe::RMulOp, jaxiteword::MulNoRelinOp>>(
        typeConverter, context);
    patterns.add<ConvertNegateOp<lwe::RNegateOp>>(typeConverter, context);
    patterns.add<ConvertNegateOp<ckks::NegateOp>>(typeConverter, context);
    patterns.add<ConvertRotateOp>(typeConverter, context);
    patterns.add<ConvertRelinOp>(typeConverter, context);
    patterns.add<ConvertModSwitchOp<bgv::ModulusSwitchOp>>(typeConverter,
                                                           context);
    patterns.add<ConvertModSwitchOp<ckks::RescaleOp>>(typeConverter, context);
    patterns.add<ConvertModSwitchOp<ckks::LevelReduceOp>>(typeConverter,
                                                          context);
    patterns.add<ConvertEncodeOp>(typeConverter, context);
    patterns.add<ConvertEncryptOp>(typeConverter, context);
    patterns.add<ConvertDecryptOp>(typeConverter, context);
    patterns.add<ConvertDecodeOp>(typeConverter, context);
    patterns.add<ConvertBinOp<lwe::RAddPlainOp, jaxiteword::AddPlainOp>>(
        typeConverter, context);
    patterns.add<ConvertBinOp<lwe::RSubPlainOp, jaxiteword::SubPlainOp>>(
        typeConverter, context);
    patterns.add<ConvertBinOp<lwe::RMulPlainOp, jaxiteword::MulPlainOp>>(
        typeConverter, context);
    patterns.add<ConvertBinOp<ckks::AddPlainOp, jaxiteword::AddPlainOp>>(
        typeConverter, context);
    patterns.add<ConvertBinOp<ckks::SubPlainOp, jaxiteword::SubPlainOp>>(
        typeConverter, context);
    patterns.add<ConvertBinOp<ckks::MulPlainOp, jaxiteword::MulPlainOp>>(
        typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

void registerLWEToJaxiteWordPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<LWEToJaxiteWord>();
  });
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
