#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_H_

#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DECL
#include "lib/Dialect/LWE/Conversions/LWEToJaxiteWord/LWEToJaxiteWord.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LWE/Conversions/LWEToJaxiteWord/LWEToJaxiteWord.h.inc"

class ToJaxiteWordTypeConverter : public TypeConverter {
 public:
  ToJaxiteWordTypeConverter(MLIRContext *ctx);
};

FailureOr<Value> getContextualCryptoContext(Operation *op);

template <typename UnaryOp, typename JaxiteWordOp>
struct ConvertUnaryOp : public OpConversionPattern<UnaryOp> {
  using OpConversionPattern<UnaryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnaryOp op, typename UnaryOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, rewriter.create<JaxiteWordOp>(
                               op.getLoc(), cryptoContext, adaptor.getInput()));
    return success();
  }
};

template <typename BinOp, typename JaxiteWordOp>
struct ConvertLWEBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<JaxiteWordOp>(op, op.getOutput().getType(),
                                              cryptoContext, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

template <typename BinOp, typename JaxiteWordOp>
struct ConvertCiphertextPlaintextOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<JaxiteWordOp>(
        op, op.getOutput().getType(), cryptoContext,
        adaptor.getCiphertextInput(), adaptor.getPlaintextInput());
    return success();
  }
};

inline bool checkRelinToBasis(llvm::ArrayRef<int> toBasis) {
  if (toBasis.size() != 2) return false;
  return toBasis[0] == 0 && toBasis[1] == 1;
}

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_H_
