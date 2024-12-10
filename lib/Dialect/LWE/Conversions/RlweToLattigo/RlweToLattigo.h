#ifndef LIB_DIALECT_LWE_CONVERSIONS_RLWETOLATTIGOUTILS_RLWETOLATTIGO_H_
#define LIB_DIALECT_LWE_CONVERSIONS_RLWETOLATTIGOUTILS_RLWETOLATTIGO_H_

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
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

template <typename Dialect, typename EvaluatorType>
struct AddEvaluatorArg : public OpConversionPattern<func::FuncOp> {
  AddEvaluatorArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsLweOrDialect<Dialect>(op)) {
      return failure();
    }

    auto evaluatorType = EvaluatorType::get(getContext());
    FunctionType originalType = op.getFunctionType();
    llvm::SmallVector<Type, 4> newTypes;
    newTypes.reserve(originalType.getNumInputs() + 1);
    newTypes.push_back(evaluatorType);
    for (auto t : originalType.getInputs()) {
      newTypes.push_back(t);
    }
    auto newFuncType =
        FunctionType::get(getContext(), newTypes, originalType.getResults());
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(newFuncType);

      Block &block = op.getBody().getBlocks().front();
      block.insertArgument(&block.getArguments().front(), evaluatorType,
                           op.getLoc());
    });

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

}  // namespace mlir::heir

#endif  // LIB_DIALECT_LWE_CONVERSIONS_RLWETOLATTIGOUTILS_RLWETOLATTIGO_H_
