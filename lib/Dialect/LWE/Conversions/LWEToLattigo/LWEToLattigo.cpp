#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
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
      (void)op.insertArguments(argIndices, selectedEvaluators, argAttrs,
                               argLocs);
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

    rewriter.modifyOpInPlace(op, [&] { (void)op.eraseArguments(argsToErase); });
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

    rewriter
        .replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes, newOperands)
        ->setDialectAttrs(op->getDialectAttrs());
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
    rewriter
        .replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes, newOperands)
        ->setDialectAttrs(op->getDialectAttrs());
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

// Lattigo API enforces ciphertext, plaintext ordering.
template <typename EvaluatorType, typename PlainOp, typename LattigoPlainOp>
struct ConvertRlweCommutativePlainOp : public OpConversionPattern<PlainOp> {
  using OpConversionPattern<PlainOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PlainOp op, typename PlainOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value ciphertext = adaptor.getLhs();
    Value plaintext = adaptor.getRhs();
    if (!isa<lattigo::RLWECiphertextType>(adaptor.getLhs().getType())) {
      ciphertext = adaptor.getRhs();
      plaintext = adaptor.getLhs();
    }

    Value evaluator = result.value();
    rewriter.replaceOpWithNewOp<LattigoPlainOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, ciphertext, plaintext);
    return success();
  }
};

// Lattigo API enforces ciphertext, plaintext ordering.
template <typename EvaluatorType, typename PlainOp, typename LattigoPlainOp,
          typename LattigoAddOp, typename LattigoNegateOp>
struct ConvertRlweSubPlainOp : public OpConversionPattern<PlainOp> {
  using OpConversionPattern<PlainOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PlainOp op, typename PlainOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    if (isa<lattigo::RLWECiphertextType>(adaptor.getLhs().getType())) {
      // Lattigo API enforces ciphertext, plaintext ordering, so we can use
      // LattigoPlainOp directly.
      Value ciphertext = adaptor.getLhs();
      Value plaintext = adaptor.getRhs();
      rewriter.replaceOpWithNewOp<LattigoPlainOp>(
          op, this->typeConverter->convertType(op.getOutput().getType()),
          evaluator, ciphertext, plaintext);
      return success();
    }

    // handle plaintext - ciphertext using (-ciphertext) + plaintext
    Value plaintext = adaptor.getLhs();
    Value ciphertext = adaptor.getRhs();

    auto negated = rewriter.create<LattigoNegateOp>(
        op.getLoc(), this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, ciphertext);
    rewriter.replaceOpWithNewOp<LattigoAddOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, negated, plaintext);
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

template <typename EvaluatorType, typename LevelReduceOp,
          typename LattigoLevelReduceOp>
struct ConvertRlweLevelReduceOp : public OpConversionPattern<LevelReduceOp> {
  using OpConversionPattern<LevelReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LevelReduceOp op, typename LevelReduceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOp(
        op, rewriter.create<LattigoLevelReduceOp>(
                op.getLoc(),
                this->typeConverter->convertType(op.getOutput().getType()),
                evaluator, adaptor.getInput(), op.getLevelToDrop()));
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

    Value input = adaptor.getInput();
    auto alloc = rewriter.create<AllocOp>(
        op.getLoc(), this->typeConverter->convertType(op.getOutput().getType()),
        params);

    auto encoding = op.getEncoding();
    int64_t scale = lwe::getScalingFactorFromEncodingAttr(encoding);

    rewriter
        .replaceOpWithNewOp<LattigoEncodeOp>(
            op, this->typeConverter->convertType(op.getOutput().getType()),
            evaluator, input, alloc, rewriter.getI64IntegerAttr(scale))
        ->setDialectAttrs(op->getDialectAttrs());
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

    // With the new layout system, decoding for RLWE will always produce a
    // tensor type.
    auto outputType = op.getOutput().getType();
    RankedTensorType outputTensorType = dyn_cast<RankedTensorType>(outputType);

    auto zeroAttr = rewriter.getZeroAttr(outputTensorType);
    if (!zeroAttr) {
      return op.emitOpError() << "Unsupported type for lowering";
    }
    auto alloc =
        rewriter.create<AllocOp>(op.getLoc(), outputTensorType, zeroAttr);

    auto decodeOp = rewriter.create<LattigoDecodeOp>(
        op.getLoc(), outputTensorType, evaluator, adaptor.getInput(), alloc);

    rewriter.replaceOp(op, decodeOp.getResult());
    return success();
  }
};

struct ConvertLWEReinterpretApplicationData
    : public OpConversionPattern<lwe::ReinterpretApplicationDataOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::ReinterpretApplicationDataOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Erase reinterpret application data.
    // If operand has no defining op, we can not replace it with defining op.
    rewriter.replaceAllOpUsesWith(op, adaptor.getOperands()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

// BGV
using ConvertBGVAddOp = ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RAddOp,
                                         lattigo::BGVAddNewOp>;
using ConvertBGVSubOp = ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RSubOp,
                                         lattigo::BGVSubNewOp>;
using ConvertBGVMulOp = ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RMulOp,
                                         lattigo::BGVMulNewOp>;
using ConvertBGVAddPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::BGVEvaluatorType, lwe::RAddPlainOp,
                                  lattigo::BGVAddNewOp>;
using ConvertBGVSubPlainOp =
    ConvertRlweSubPlainOp<lattigo::BGVEvaluatorType, lwe::RSubPlainOp,
                          lattigo::BGVSubNewOp, lattigo::BGVAddNewOp,
                          lattigo::RLWENegateNewOp>;
using ConvertBGVMulPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::BGVEvaluatorType, lwe::RMulPlainOp,
                                  lattigo::BGVMulNewOp>;

using ConvertBGVRelinOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, bgv::RelinearizeOp,
                       lattigo::BGVRelinearizeNewOp>;
using ConvertBGVModulusSwitchOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, bgv::ModulusSwitchOp,
                       lattigo::BGVRescaleNewOp>;

using ConvertBGVRotateColumnsOp =
    ConvertRlweRotateOp<lattigo::BGVEvaluatorType, bgv::RotateColumnsOp,
                        lattigo::BGVRotateColumnsNewOp>;

using ConvertBGVEncryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEEncryptorType, lwe::RLWEEncryptOp,
                       lattigo::RLWEEncryptOp>;
using ConvertBGVDecryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEDecryptorType, lwe::RLWEDecryptOp,
                       lattigo::RLWEDecryptOp>;
using ConvertBGVEncodeOp =
    ConvertRlweEncodeOp<lattigo::BGVEncoderType, lattigo::BGVParameterType,
                        lwe::RLWEEncodeOp, lattigo::BGVEncodeOp,
                        lattigo::BGVNewPlaintextOp>;
using ConvertBGVDecodeOp =
    ConvertRlweDecodeOp<lattigo::BGVEncoderType, lwe::RLWEDecodeOp,
                        lattigo::BGVDecodeOp, arith::ConstantOp>;

using ConvertBGVLevelReduceOp =
    ConvertRlweLevelReduceOp<lattigo::BGVEvaluatorType, bgv::LevelReduceOp,
                             lattigo::RLWEDropLevelNewOp>;

// CKKS
using ConvertCKKSAddOp = ConvertRlweBinOp<lattigo::CKKSEvaluatorType,
                                          lwe::RAddOp, lattigo::CKKSAddNewOp>;
using ConvertCKKSSubOp = ConvertRlweBinOp<lattigo::CKKSEvaluatorType,
                                          lwe::RSubOp, lattigo::CKKSSubNewOp>;
using ConvertCKKSMulOp = ConvertRlweBinOp<lattigo::CKKSEvaluatorType,
                                          lwe::RMulOp, lattigo::CKKSMulNewOp>;
using ConvertCKKSAddPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::CKKSEvaluatorType, lwe::RAddPlainOp,
                                  lattigo::CKKSAddNewOp>;
using ConvertCKKSSubPlainOp =
    ConvertRlweSubPlainOp<lattigo::CKKSEvaluatorType, lwe::RSubPlainOp,
                          lattigo::CKKSSubNewOp, lattigo::CKKSAddNewOp,
                          lattigo::RLWENegateNewOp>;
using ConvertCKKSMulPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::CKKSEvaluatorType, lwe::RMulPlainOp,
                                  lattigo::CKKSMulNewOp>;

using ConvertCKKSRelinOp =
    ConvertRlweUnaryOp<lattigo::CKKSEvaluatorType, ckks::RelinearizeOp,
                       lattigo::CKKSRelinearizeNewOp>;
using ConvertCKKSModulusSwitchOp =
    ConvertRlweUnaryOp<lattigo::CKKSEvaluatorType, ckks::RescaleOp,
                       lattigo::CKKSRescaleNewOp>;

using ConvertCKKSRotateOp =
    ConvertRlweRotateOp<lattigo::CKKSEvaluatorType, ckks::RotateOp,
                        lattigo::CKKSRotateNewOp>;

using ConvertCKKSEncryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEEncryptorType, lwe::RLWEEncryptOp,
                       lattigo::RLWEEncryptOp>;
using ConvertCKKSDecryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEDecryptorType, lwe::RLWEDecryptOp,
                       lattigo::RLWEDecryptOp>;
using ConvertCKKSEncodeOp =
    ConvertRlweEncodeOp<lattigo::CKKSEncoderType, lattigo::CKKSParameterType,
                        lwe::RLWEEncodeOp, lattigo::CKKSEncodeOp,
                        lattigo::CKKSNewPlaintextOp>;
using ConvertCKKSDecodeOp =
    ConvertRlweDecodeOp<lattigo::CKKSEncoderType, lwe::RLWEDecodeOp,
                        lattigo::CKKSDecodeOp, arith::ConstantOp>;

using ConvertCKKSLevelReduceOp =
    ConvertRlweLevelReduceOp<lattigo::CKKSEvaluatorType, ckks::LevelReduceOp,
                             lattigo::RLWEDropLevelNewOp>;

#define GEN_PASS_DEF_LWETOLATTIGO
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h.inc"

struct LWEToLattigo : public impl::LWEToLattigoBase<LWEToLattigo> {
  // See https://github.com/llvm/llvm-project/pull/127772
  // During dialect conversion, the attribute of the func::CallOp is not
  // preserved. We save the dialect attributes of func::CallOp before
  // conversion and restore them after conversion.
  //
  // Note that this is not safe as after conversion the order of func::CallOp
  // may change. However, this is the best we can do for now as we do not have
  // a map from the old func::CallOp to the new func::CallOp.
  SmallVector<SmallVector<NamedAttribute>> funcCallOpDialectAttrs;

  void saveFuncCallOpDialectAttrs() {
    funcCallOpDialectAttrs.clear();
    auto *module = getOperation();
    module->walk([&](func::CallOp callOp) {
      SmallVector<NamedAttribute> dialectAttrs;
      for (auto namedAttr : callOp->getDialectAttrs()) {
        dialectAttrs.push_back(namedAttr);
      }
      funcCallOpDialectAttrs.push_back(dialectAttrs);
    });
  }

  void restoreFuncCallOpDialectAttrs() {
    auto *module = getOperation();
    auto *funcCallOpDialectAttrsIter = funcCallOpDialectAttrs.begin();
    module->walk([&](func::CallOp callOp) {
      callOp->setDialectAttrs(*funcCallOpDialectAttrsIter);
      ++funcCallOpDialectAttrsIter;
    });
  }

  void runOnOperation() override {
    // Save the dialect attributes of func::CallOp before conversion.
    saveFuncCallOpDialectAttrs();

    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToLattigoTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<lattigo::LattigoDialect>();
    target.addIllegalDialect<bgv::BGVDialect, ckks::CKKSDialect>();
    target
        .addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp, lwe::RLWEEncodeOp,
                      lwe::RLWEDecodeOp, lwe::RAddOp, lwe::RSubOp, lwe::RMulOp,
                      lwe::RMulPlainOp, lwe::RSubPlainOp, lwe::RAddPlainOp,
                      lwe::ReinterpretApplicationDataOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg =
          op.getFunctionType().getNumInputs() > 0 &&
          containsArgumentOfType<
              lattigo::BGVEvaluatorType, lattigo::BGVEncoderType,
              lattigo::CKKSEvaluatorType, lattigo::CKKSEncoderType,
              lattigo::RLWEEncryptorType, lattigo::RLWEDecryptorType>(op);

      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect,
                                         ckks::CKKSDialect>(op) ||
              hasCryptoContextArg);
    });

    // Ensures that callee function signature is consistent
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      auto operandTypes = op.getCalleeType().getInputs();
      auto containsCryptoArg = llvm::any_of(operandTypes, [&](Type argType) {
        return DialectEqual<lwe::LWEDialect, bgv::BGVDialect, ckks::CKKSDialect,
                            lattigo::LattigoDialect>()(&argType.getDialect());
      });
      auto hasCryptoContextArg =
          !operandTypes.empty() &&
          mlir::isa<lattigo::BGVEvaluatorType, lattigo::CKKSEvaluatorType>(
              *operandTypes.begin());
      return (!containsCryptoArg || hasCryptoContextArg);
    });

    OpPredicate containsEncryptUseSk = [&](Operation *op) -> bool {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        // for declaration, assume its uses are decrypt
        if (funcOp.isDeclaration()) {
          return false;
        }
        return llvm::any_of(funcOp.getArguments(), [&](BlockArgument arg) {
          return mlir::isa<lwe::NewLWESecretKeyType>(arg.getType()) &&
                 llvm::any_of(arg.getUses(), [&](OpOperand &use) {
                   return mlir::isa<lwe::RLWEEncryptOp>(use.getOwner());
                 });
        });
      }
      return false;
    };

    auto gateByBGVModuleAttr =
        [&](const OpPredicate &inputPredicate) -> OpPredicate {
      return [module, inputPredicate](Operation *op) {
        return moduleIsBGVOrBFV(module) && inputPredicate(op);
      };
    };

    auto gateByCKKSModuleAttr =
        [&](const OpPredicate &inputPredicate) -> OpPredicate {
      return [module, inputPredicate](Operation *op) {
        return moduleIsCKKS(module) && inputPredicate(op);
      };
    };

    OpPredicate containsNoEncryptUseSk = [&](Operation *op) -> bool {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        bool findKey =
            llvm::any_of(funcOp.getArgumentTypes(), [&](Type argType) {
              return mlir::isa<lwe::NewLWESecretKeyType>(argType);
            });
        // for declaration, only checks the existence
        if (funcOp.isDeclaration()) {
          return findKey;
        }
        // for definition, check the uses
        bool noEncrypt =
            llvm::all_of(funcOp.getArguments(), [&](BlockArgument arg) {
              return !mlir::isa<lwe::NewLWESecretKeyType>(arg.getType()) ||
                     llvm::none_of(arg.getUses(), [&](OpOperand &use) {
                       return mlir::isa<lwe::RLWEEncryptOp>(use.getOwner());
                     });
            });
        return findKey && noEncrypt;
      }
      return false;
    };

    std::vector<std::pair<Type, OpPredicate>> evaluators;

    // param/encoder also needed for the main func
    // as there might (not) be ct-pt operations
    evaluators = {
        {lattigo::BGVEvaluatorType::get(context),
         gateByBGVModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>)},
        {lattigo::BGVParameterType::get(context),
         gateByBGVModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>)},
        {lattigo::BGVEncoderType::get(context),
         gateByBGVModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>)},
        {lattigo::CKKSEvaluatorType::get(context),
         gateByCKKSModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>)},
        {lattigo::CKKSParameterType::get(context),
         gateByCKKSModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>)},
        {lattigo::CKKSEncoderType::get(context),
         gateByCKKSModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>)},
        {lattigo::RLWEEncryptorType::get(context, /*publicKey*/ true),
         containsArgumentOfType<lwe::NewLWEPublicKeyType>},
        // for NewLWESecretKey, if its uses are encrypt, then convert it to an
        // encryptor, otherwise, convert it to a decryptor
        {lattigo::RLWEEncryptorType::get(context, /*publicKey*/ false),
         containsEncryptUseSk},
        {lattigo::RLWEDecryptorType::get(context), containsNoEncryptUseSk},
    };

    patterns.add<AddEvaluatorArg>(context, evaluators);
    patterns.add<ConvertFuncCallOp>(context, evaluators);

    if (moduleIsBGVOrBFV(module)) {
      patterns.add<ConvertBGVAddOp, ConvertBGVSubOp, ConvertBGVMulOp,
                   ConvertBGVAddPlainOp, ConvertBGVSubPlainOp,
                   ConvertBGVMulPlainOp, ConvertBGVRelinOp,
                   ConvertBGVModulusSwitchOp, ConvertBGVRotateColumnsOp,
                   ConvertBGVEncryptOp, ConvertBGVDecryptOp, ConvertBGVEncodeOp,
                   ConvertBGVDecodeOp, ConvertBGVLevelReduceOp>(typeConverter,
                                                                context);
    }
    if (moduleIsCKKS(module)) {
      patterns.add<
          ConvertCKKSAddOp, ConvertCKKSSubOp, ConvertCKKSMulOp,
          ConvertCKKSAddPlainOp, ConvertCKKSSubPlainOp, ConvertCKKSMulPlainOp,
          ConvertCKKSRelinOp, ConvertCKKSModulusSwitchOp, ConvertCKKSRotateOp,
          ConvertCKKSEncryptOp, ConvertCKKSDecryptOp, ConvertCKKSEncodeOp,
          ConvertCKKSDecodeOp, ConvertCKKSLevelReduceOp>(typeConverter,
                                                         context);
    }
    // Misc
    patterns.add<ConvertLWEReinterpretApplicationData>(typeConverter, context);

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

    // Restore the dialect attributes of func::CallOp after conversion.
    restoreFuncCallOpDialectAttrs();
  }
};

}  // namespace mlir::heir::lwe
