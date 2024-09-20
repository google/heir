#ifndef LIB_CONVERSION_LWETOOPENFHE_LWETOOPENFHE_H_
#define LIB_CONVERSION_LWETOOPENFHE_LWETOOPENFHE_H_

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "mlir/include/mlir/Pass/Pass.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

struct ConvertEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  ConvertEncryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::RLWEEncryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

struct ConvertDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  ConvertDecryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::RLWEDecryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(lwe::RLWEDecryptOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const;
};

struct ConvertEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  ConvertEncodeOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::RLWEEncodeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // OpenFHE has a convention that all inputs to MakePackedPlaintext are
  // std::vector<int64_t>, so we need to cast the input to that type.
  LogicalResult matchAndRewrite(lwe::RLWEEncodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const;
};

}  // namespace mlir::heir::lwe

#endif  // LIB_CONVERSION_LWETOOPENFHE_LWETOOPENFHE_H_
