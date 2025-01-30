#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h"

#include <cassert>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::lwe {

class ToLattigoTypeConverter : public TypeConverter {
 public:
  ToLattigoTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::NewLWECiphertextType type) -> Type {
      return lattigo::RLWECiphertextType::get(ctx);
    });
    addConversion([ctx](lwe::NewLWEPlaintextType type) -> Type {
      return lattigo::RLWEPlaintextType::get(ctx);
    });
    addConversion([ctx](lwe::NewLWEPublicKeyType type) -> Type {
      return lattigo::RLWEPublicKeyType::get(ctx);
    });
    addConversion([ctx](lwe::NewLWESecretKeyType type) -> Type {
      return lattigo::RLWESecretKeyType::get(ctx);
    });
  }
};

namespace {
template <typename EvaluatorType>
FailureOr<Value> getContextualEvaluator(Operation *op) {
  auto result = getContextualArgFromFunc<EvaluatorType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found RLWE op in a function without a public "
              "key argument. Did the AddEvaluatorArg pattern fail to run?";
  }
  return result.value();
}

FailureOr<Value> getContextualEvaluator(Operation *op, Type type) {
  return getContextualArgFromFunc(op, type);
}

// NOTE: we can not use containsDialect
// for FuncOp declaration, which does not have a body
template <typename... Dialects>
bool containsArgumentOfDialect(Operation *op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    return false;
  }
  return llvm::any_of(funcOp.getArgumentTypes(), [&](Type argType) {
    return DialectEqual<Dialects...>()(&argType.getDialect());
  });
}

struct AddEvaluatorArg : public OpConversionPattern<func::FuncOp> {
  AddEvaluatorArg(mlir::MLIRContext *context,
                  const std::vector<std::pair<Type, OpPredicate>> &evaluators)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2),
        evaluators(evaluators) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 4> selectedEvaluators;

    for (const auto &evaluator : evaluators) {
      auto predicate = evaluator.second;
      if (predicate(op)) {
        selectedEvaluators.push_back(evaluator.first);
      }
    }

    if (selectedEvaluators.empty()) {
      return success();
    }

    // Insert all argument at the beginning
    // NOTE: arguments with identical index will
    // appear in the same order that they were listed.
    SmallVector<unsigned> argIndices(selectedEvaluators.size(), 0);
    SmallVector<DictionaryAttr> argAttrs(selectedEvaluators.size(), nullptr);
    SmallVector<Location> argLocs(selectedEvaluators.size(), op.getLoc());

    rewriter.modifyOpInPlace(op, [&] {
      SmallVector<unsigned> argIndices(selectedEvaluators.size(), 0);
      if (op.isDeclaration()) {
        auto newFuncType = op.getTypeWithArgsAndResults(
            argIndices, selectedEvaluators, {}, {});
        op.setType(newFuncType);
      } else {
        op.insertArguments(argIndices, selectedEvaluators, argAttrs, argLocs);
      }
    });
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

template <typename KeyType>
struct RemoveKeyArg : public OpConversionPattern<func::FuncOp> {
  RemoveKeyArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ::llvm::BitVector argsToErase(op.getNumArguments());

    for (auto i = 0; i != op.getNumArguments(); ++i) {
      if (mlir::isa<KeyType>(op.getArgumentTypes()[i])) {
        argsToErase.set(i);
      }
    }

    if (argsToErase.none()) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&] {
      if (op.isDeclaration()) {
        // without for deletion
        auto newFuncType = op.getTypeWithoutArgsAndResults(
            argsToErase, /* resultIndices= */ BitVector(op->getNumResults()));
        op.setType(newFuncType);
      } else {
        op.eraseArguments(argsToErase);
      }
    });
    return success();
  }
};

struct ConvertFuncCallOp : public OpConversionPattern<func::CallOp> {
  ConvertFuncCallOp(mlir::MLIRContext *context,
                    const std::vector<std::pair<Type, OpPredicate>> &evaluators)
      : OpConversionPattern<func::CallOp>(context), evaluators(evaluators) {}

  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> selectedevaluatorsValues;
    for (const auto &evaluator : evaluators) {
      auto result = getContextualEvaluator(op.getOperation(), evaluator.first);
      // filter out non-existent evaluators
      if (failed(result)) {
        continue;
      }
      selectedevaluatorsValues.push_back(result.value());
    }

    auto callee = op.getCallee();
    auto operands = adaptor.getOperands();
    auto resultTypes = op.getResultTypes();

    SmallVector<Value> newOperands;
    for (auto evaluator : selectedevaluatorsValues) {
      newOperands.push_back(evaluator);
    }
    for (auto operand : operands) {
      newOperands.push_back(operand);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes,
                                              newOperands);
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

template <typename KeyType>
struct RemoveKeyArgForFuncCall : public OpConversionPattern<func::CallOp> {
  RemoveKeyArgForFuncCall(mlir::MLIRContext *context)
      : OpConversionPattern<func::CallOp>(context) {}

  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    auto operands = adaptor.getOperands();
    auto resultTypes = op.getResultTypes();

    SmallVector<Value> newOperands;
    for (auto operand : operands) {
      if (!mlir::isa<KeyType>(operand.getType())) {
        newOperands.push_back(operand);
      }
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes,
                                              newOperands);
    return success();
  }
};

template <typename EvaluatorType, typename UnaryOp, typename LattigoUnaryOp>
struct ConvertRlweUnaryOp : public OpConversionPattern<UnaryOp> {
  using OpConversionPattern<UnaryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnaryOp op, typename UnaryOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOp(
        op, rewriter.create<LattigoUnaryOp>(
                op.getLoc(),
                this->typeConverter->convertType(op.getOutput().getType()),
                evaluator, adaptor.getInput()));
    return success();
  }
};

template <typename EvaluatorType, typename BinOp, typename LattigoBinOp>
struct ConvertRlweBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOpWithNewOp<LattigoBinOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

template <typename EvaluatorType, typename PlainOp, typename LattigoPlainOp>
struct ConvertRlwePlainOp : public OpConversionPattern<PlainOp> {
  using OpConversionPattern<PlainOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PlainOp op, typename PlainOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOpWithNewOp<LattigoPlainOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, adaptor.getCiphertextInput(), adaptor.getPlaintextInput());
    return success();
  }
};

template <typename EvaluatorType, typename RlweRotateOp,
          typename LattigoRotateOp>
struct ConvertRlweRotateOp : public OpConversionPattern<RlweRotateOp> {
  ConvertRlweRotateOp(mlir::MLIRContext *context)
      : OpConversionPattern<RlweRotateOp>(context) {}

  using OpConversionPattern<RlweRotateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RlweRotateOp op, typename RlweRotateOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOp(
        op, rewriter.create<LattigoRotateOp>(
                op.getLoc(),
                this->typeConverter->convertType(op.getOutput().getType()),
                evaluator, adaptor.getInput(), adaptor.getOffset()));
    return success();
  }
};

template <typename EvaluatorType, typename ParamType, typename EncodeOp,
          typename LattigoEncodeOp, typename AllocOp>
struct ConvertRlweEncodeOp : public OpConversionPattern<EncodeOp> {
  using OpConversionPattern<EncodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EncodeOp op, typename EncodeOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;
    Value evaluator = result.value();

    FailureOr<Value> result2 =
        getContextualEvaluator<ParamType>(op.getOperation());
    if (failed(result2)) return result2;
    Value params = result2.value();

    auto alloc = rewriter.create<AllocOp>(
        op.getLoc(), this->typeConverter->convertType(op.getOutput().getType()),
        params);

    rewriter.replaceOpWithNewOp<LattigoEncodeOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, adaptor.getInput(), alloc);
    return success();
  }
};

template <typename EvaluatorType, typename DecodeOp, typename LattigoDecodeOp,
          typename AllocOp>
struct ConvertRlweDecodeOp : public OpConversionPattern<DecodeOp> {
  using OpConversionPattern<DecodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DecodeOp op, typename DecodeOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;
    Value evaluator = result.value();

    auto outputType = op.getOutput().getType();
    RankedTensorType outputTensorType = dyn_cast<RankedTensorType>(outputType);
    bool isScalar = false;
    if (!outputTensorType) {
      isScalar = true;
      outputTensorType = RankedTensorType::get({1}, outputType);
    }

    APInt zero(getElementTypeOrSelf(outputType).getIntOrFloatBitWidth(), 0);

    auto constant = DenseElementsAttr::get(outputTensorType, zero);

    auto alloc =
        rewriter.create<AllocOp>(op.getLoc(), outputTensorType, constant);

    auto decodeOp = rewriter.create<LattigoDecodeOp>(
        op.getLoc(), outputTensorType, evaluator, adaptor.getInput(), alloc);

    // TODO(#1174): the sin of lwe.reinterpret_underlying_type
    if (isScalar) {
      SmallVector<Value, 1> indices;
      auto index = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                      rewriter.getIndexAttr(0));
      indices.push_back(index);
      auto extract = rewriter.create<tensor::ExtractOp>(
          op.getLoc(), decodeOp.getResult(), indices);
      rewriter.replaceOp(op, extract.getResult());
    } else {
      rewriter.replaceOp(op, decodeOp.getResult());
    }
    return success();
  }
};

struct ConvertLWEReinterpretUnderlyingType
    : public OpConversionPattern<lwe::ReinterpretUnderlyingTypeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::ReinterpretUnderlyingTypeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // erase reinterpret underlying
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

}  // namespace

using ConvertAddOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RAddOp, lattigo::BGVAddOp>;
using ConvertSubOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RSubOp, lattigo::BGVSubOp>;
using ConvertMulOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RMulOp, lattigo::BGVMulOp>;
using ConvertAddPlainOp =
    ConvertRlwePlainOp<lattigo::BGVEvaluatorType, bgv::AddPlainOp,
                       lattigo::BGVAddOp>;
using ConvertSubPlainOp =
    ConvertRlwePlainOp<lattigo::BGVEvaluatorType, bgv::SubPlainOp,
                       lattigo::BGVSubOp>;
using ConvertMulPlainOp =
    ConvertRlwePlainOp<lattigo::BGVEvaluatorType, bgv::MulPlainOp,
                       lattigo::BGVMulOp>;

using ConvertRelinOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, bgv::RelinearizeOp,
                       lattigo::BGVRelinearizeOp>;
using ConvertModulusSwitchOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, bgv::ModulusSwitchOp,
                       lattigo::BGVRescaleOp>;

// TODO(#1186): figure out generic rotating using BGVRotateColumns/RowsOp
using ConvertRotateOp =
    ConvertRlweRotateOp<lattigo::BGVEvaluatorType, bgv::RotateOp,
                        lattigo::BGVRotateColumnsOp>;

using ConvertEncryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEEncryptorType, lwe::RLWEEncryptOp,
                       lattigo::RLWEEncryptOp>;
using ConvertDecryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEDecryptorType, lwe::RLWEDecryptOp,
                       lattigo::RLWEDecryptOp>;
using ConvertEncodeOp =
    ConvertRlweEncodeOp<lattigo::BGVEncoderType, lattigo::BGVParameterType,
                        lwe::RLWEEncodeOp, lattigo::BGVEncodeOp,
                        lattigo::BGVNewPlaintextOp>;
using ConvertDecodeOp =
    ConvertRlweDecodeOp<lattigo::BGVEncoderType, lwe::RLWEDecodeOp,
                        lattigo::BGVDecodeOp, arith::ConstantOp>;

#define GEN_PASS_DEF_LWETOLATTIGO
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h.inc"

struct LWEToLattigo : public impl::LWEToLattigoBase<LWEToLattigo> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToLattigoTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<lattigo::LattigoDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target
        .addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp, lwe::RLWEEncodeOp,
                      lwe::RLWEDecodeOp, lwe::RAddOp, lwe::RSubOp, lwe::RMulOp,
                      lwe::ReinterpretUnderlyingTypeOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg =
          op.getFunctionType().getNumInputs() > 0 &&
          containsArgumentOfType<
              lattigo::BGVEvaluatorType, lattigo::BGVEncoderType,
              lattigo::RLWEEncryptorType, lattigo::RLWEDecryptorType>(op);

      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>(
                  op) ||
              hasCryptoContextArg);
    });

    // Ensures that callee function signature is consistent
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      auto operandTypes = op.getCalleeType().getInputs();
      auto containsCryptoArg = llvm::any_of(operandTypes, [&](Type argType) {
        return DialectEqual<lwe::LWEDialect, bgv::BGVDialect,
                            lattigo::LattigoDialect>()(&argType.getDialect());
      });
      auto hasCryptoContextArg =
          !operandTypes.empty() &&
          mlir::isa<lattigo::BGVEvaluatorType>(*operandTypes.begin());
      return (!containsCryptoArg || hasCryptoContextArg);
    });

    std::vector<std::pair<Type, OpPredicate>> evaluators;

    // param/encoder also needed for the main func
    // as there might (not) be ct-pt operations
    evaluators = {
        {lattigo::BGVEvaluatorType::get(context),
         containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>},
        {lattigo::BGVParameterType::get(context),
         containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>},
        {lattigo::BGVEncoderType::get(context),
         containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>},
        {lattigo::RLWEEncryptorType::get(context),
         containsArgumentOfType<lwe::NewLWEPublicKeyType>},
        {lattigo::RLWEDecryptorType::get(context),
         containsArgumentOfType<lwe::NewLWESecretKeyType>},
    };

    patterns.add<AddEvaluatorArg>(context, evaluators);
    patterns.add<ConvertFuncCallOp>(context, evaluators);

    patterns.add<ConvertAddOp, ConvertSubOp, ConvertMulOp, ConvertAddPlainOp,
                 ConvertSubPlainOp, ConvertMulPlainOp, ConvertRelinOp,
                 ConvertModulusSwitchOp, ConvertRotateOp, ConvertEncryptOp,
                 ConvertDecryptOp, ConvertEncodeOp, ConvertDecodeOp,
                 ConvertLWEReinterpretUnderlyingType>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // remove key args from function calls
    // walkAndApplyPatterns will cause segfault at MLIR side
    RewritePatternSet postPatterns(context);
    postPatterns.add<RemoveKeyArgForFuncCall<lattigo::RLWESecretKeyType>>(
        context);
    postPatterns.add<RemoveKeyArgForFuncCall<lattigo::RLWEPublicKeyType>>(
        context);

    ConversionTarget postTarget(*context);
    postTarget.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return llvm::none_of(op.getOperandTypes(), [&](Type operandType) {
        return mlir::isa<lattigo::RLWESecretKeyType,
                         lattigo::RLWEPublicKeyType>(operandType);
      });
    });
    if (failed(applyPartialConversion(module, postTarget,
                                      std::move(postPatterns)))) {
      return signalPassFailure();
    }

    // remove unused key args from function types
    // in favor of encryptor/decryptor
    RewritePatternSet postPatterns2(context);
    postPatterns2.add<RemoveKeyArg<lattigo::RLWESecretKeyType>>(context);
    postPatterns2.add<RemoveKeyArg<lattigo::RLWEPublicKeyType>>(context);
    walkAndApplyPatterns(module, std::move(postPatterns2));
  }
};

}  // namespace mlir::heir::lwe
