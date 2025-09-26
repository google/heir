#ifndef LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_
#define LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Lower a client encryption function's secret.conceal op to lwe.rlwe_encode +
// lwe.rlwe_encrypt. Modifies the containing function to add new secret key
// material args.
// TODO(#1875): support trivial encryptions
struct ConvertClientConceal : public OpConversionPattern<secret::ConcealOp> {
  ConvertClientConceal(const TypeConverter& typeConverter_,
                       mlir::MLIRContext* context, bool usePublicKey,
                       polynomial::RingAttr ring)
      : OpConversionPattern<secret::ConcealOp>(typeConverter_, context),
        usePublicKey(usePublicKey),
        ring(ring) {}

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;

 private:
  bool usePublicKey;
  polynomial::RingAttr ring;
};

// Lower a client decryption function's secret.reveal op to lwe.rlwe_decrypt +
// lwe.rlwe_decode. Modifies the containing function to add new secret key
// material args.
struct ConvertClientReveal : public OpConversionPattern<secret::RevealOp> {
  ConvertClientReveal(const TypeConverter& typeConverter_,
                      mlir::MLIRContext* context, polynomial::RingAttr ring)
      : OpConversionPattern<secret::RevealOp>(typeConverter_, context),
        ring(ring) {}

  LogicalResult matchAndRewrite(
      secret::RevealOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;

 private:
  polynomial::RingAttr ring;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_
