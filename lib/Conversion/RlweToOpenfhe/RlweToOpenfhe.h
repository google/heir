#ifndef LIB_CONVERSION_RLWETOOPENFHEUTILS_RLWETOOPENFHE_H_
#define LIB_CONVERSION_RLWETOOPENFHEUTILS_RLWETOOPENFHE_H_

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

class ToOpenfheTypeConverter : public TypeConverter {
 public:
  ToOpenfheTypeConverter(MLIRContext *ctx);
};

FailureOr<Value> getContextualCryptoContext(Operation *op);

template <typename Dialect>
struct AddCryptoContextArg : public OpConversionPattern<func::FuncOp> {
  AddCryptoContextArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsLweOrDialect<Dialect>(op)) {
      return failure();
    }

    auto cryptoContextType = openfhe::CryptoContextType::get(getContext());
    FunctionType originalType = op.getFunctionType();
    llvm::SmallVector<Type, 4> newTypes;
    newTypes.reserve(originalType.getNumInputs() + 1);
    newTypes.push_back(cryptoContextType);
    for (auto t : originalType.getInputs()) {
      newTypes.push_back(t);
    }
    auto newFuncType =
        FunctionType::get(getContext(), newTypes, originalType.getResults());
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(newFuncType);

      Block &block = op.getBody().getBlocks().front();
      block.insertArgument(&block.getArguments().front(), cryptoContextType,
                           op.getLoc());
    });

    return success();
  }
};

template <typename UnaryOp, typename OpenfheUnaryOp>
struct ConvertRlweUnaryOp : public OpConversionPattern<UnaryOp> {
  using OpConversionPattern<UnaryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnaryOp op, typename UnaryOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, rewriter.create<OpenfheUnaryOp>(
                               op.getLoc(), cryptoContext, adaptor.getInput()));
    return success();
  }
};

template <typename BinOp, typename OpenfheBinOp>
struct ConvertRlweBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<OpenfheBinOp>(op, op.getOutput().getType(),
                                              cryptoContext, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

template <typename BinOp, typename OpenfheBinOp>
struct ConvertRlweCiphertextPlaintextOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<OpenfheBinOp>(
        op, op.getOutput().getType(), cryptoContext,
        adaptor.getCiphertextInput(), adaptor.getPlaintextInput());
    return success();
  }
};

template <typename RlweRotateOp>
struct ConvertRlweRotateOp : public OpConversionPattern<RlweRotateOp> {
  ConvertRlweRotateOp(mlir::MLIRContext *context)
      : OpConversionPattern<RlweRotateOp>(context) {}

  using OpConversionPattern<RlweRotateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RlweRotateOp op, typename RlweRotateOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, rewriter.create<openfhe::RotOp>(
                               op.getLoc(), cryptoContext, adaptor.getInput(),
                               adaptor.getOffset()));
    return success();
  }
};

inline bool checkRelinToBasis(llvm::ArrayRef<int> toBasis) {
  if (toBasis.size() != 2) return false;
  return toBasis[0] == 0 && toBasis[1] == 1;
}

template <typename RlweRelinOp>
struct ConvertRlweRelinOp : public OpConversionPattern<RlweRelinOp> {
  ConvertRlweRelinOp(mlir::MLIRContext *context)
      : OpConversionPattern<RlweRelinOp>(context) {}

  using OpConversionPattern<RlweRelinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RlweRelinOp op, typename RlweRelinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    auto toBasis = adaptor.getToBasis();

    // Since the `Relinearize()` function in OpenFHE relinearizes a ciphertext
    // to the lowest level (for (1,s)), the `to_basis` of `CKKS.RelinOp` must be
    // [0,1].
    if (!checkRelinToBasis(toBasis)) {
      op.emitError() << "toBasis must be [0, 1], got [" << toBasis << "]";
      return failure();
    }

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<openfhe::RelinOp>(
        op, op.getOutput().getType(), cryptoContext, adaptor.getInput());
    return success();
  }
};

}  // namespace mlir::heir

#endif  // LIB_CONVERSION_RLWETOOPENFHEUTILS_RLWETOOPENFHE_H_
