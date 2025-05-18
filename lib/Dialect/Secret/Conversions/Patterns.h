#ifndef LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_
#define LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Lower a client encryption function's secret.conceal op to lwe.rlwe_encode +
// lwe.rlwe_encrypt. Modifies the containing function to add new secret key
// material args.
struct ConvertClientConceal
    : public ContextAwareOpConversionPattern<secret::ConcealOp> {
  ConvertClientConceal(const ContextAwareTypeConverter &typeConverter,
                       mlir::MLIRContext *context, bool usePublicKey,
                       polynomial::RingAttr ring)
      : ContextAwareOpConversionPattern<secret::ConcealOp>(typeConverter,
                                                           context),
        usePublicKey(usePublicKey),
        ring(ring) {}

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const override;

 private:
  bool usePublicKey;
  polynomial::RingAttr ring;
};

// Lower a client decryption function's secret.reveal op to lwe.rlwe_decrypt +
// lwe.rlwe_decode. Modifies the containing function to add new secret key
// material args.
struct ConvertClientReveal
    : public ContextAwareOpConversionPattern<secret::RevealOp> {
  ConvertClientReveal(const ContextAwareTypeConverter &typeConverter,
                      mlir::MLIRContext *context, polynomial::RingAttr ring)
      : ContextAwareOpConversionPattern<secret::RevealOp>(typeConverter,
                                                          context),
        ring(ring) {}

  LogicalResult matchAndRewrite(
      secret::RevealOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const override;

 private:
  polynomial::RingAttr ring;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_
