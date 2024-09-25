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

  LogicalResult matchAndRewrite(
      lwe::RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;
};

// ConvertEncodeOp takes a boolean parameter indicating whether the
// MakeCKKSPackedPlaintext should be used over the regular MakePackedPlaintext.
struct ConvertEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  explicit ConvertEncodeOp(const mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context, bool ckks)
      : mlir::OpConversionPattern<lwe::RLWEEncodeOp>(typeConverter, context),
        ckks_(ckks) {}

  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;

 private:
  bool ckks_;
};

}  // namespace mlir::heir::lwe

#endif  // LIB_CONVERSION_LWETOOPENFHE_LWETOOPENFHE_H_
