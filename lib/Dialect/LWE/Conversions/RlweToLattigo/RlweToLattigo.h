#ifndef LIB_DIALECT_LWE_CONVERSIONS_RLWETOLATTIGOUTILS_RLWETOLATTIGO_H_
#define LIB_DIALECT_LWE_CONVERSIONS_RLWETOLATTIGOUTILS_RLWETOLATTIGO_H_

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

class ToLattigoTypeConverter : public TypeConverter {
 public:
  ToLattigoTypeConverter(MLIRContext *ctx);
};

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

    FunctionType originalType = op.getFunctionType();
    llvm::SmallVector<Type, 4> newTypes;
    newTypes.reserve(originalType.getNumInputs() + selectedEvaluators.size());
    for (auto evaluatorType : selectedEvaluators) {
      newTypes.push_back(evaluatorType);
    }
    for (auto t : originalType.getInputs()) {
      newTypes.push_back(t);
    }
    auto newFuncType =
        FunctionType::get(getContext(), newTypes, originalType.getResults());
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(newFuncType);

      Block &block = op.getBody().getBlocks().front();
      for (auto evaluatorType : llvm::reverse(selectedEvaluators)) {
        block.insertArgument(&block.getArguments().front(), evaluatorType,
                             op.getLoc());
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
    SmallVector<int, 1> keyArgIndices;
    Block &block = op.getBody().getBlocks().front();
    for (auto arg : block.getArguments()) {
      if (mlir::isa<KeyType>(arg.getType()) && arg.getUses().empty()) {
        keyArgIndices.push_back(arg.getArgNumber());
      }
    }

    if (keyArgIndices.empty()) {
      return success();
    }

    FunctionType originalType = op.getFunctionType();
    llvm::SmallVector<Type, 4> newTypes;
    newTypes.reserve(originalType.getNumInputs());
    for (auto arg : block.getArguments()) {
      if (llvm::is_contained(keyArgIndices, arg.getArgNumber())) {
        continue;
      }
      newTypes.push_back(arg.getType());
    }
    auto newFuncType =
        FunctionType::get(getContext(), newTypes, originalType.getResults());
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(newFuncType);

      Block &block = op.getBody().getBlocks().front();
      for (auto arg : block.getArguments()) {
        if (llvm::is_contained(keyArgIndices, arg.getArgNumber())) {
          block.eraseArgument(arg.getArgNumber());
        }
      }
    });
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

}  // namespace mlir::heir

#endif  // LIB_DIALECT_LWE_CONVERSIONS_RLWETOLATTIGOUTILS_RLWETOLATTIGO_H_
