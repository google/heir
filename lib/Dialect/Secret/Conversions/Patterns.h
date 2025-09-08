#ifndef LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_
#define LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_

#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
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
struct ConvertClientConceal
    : public ContextAwareOpConversionPattern<secret::ConcealOp> {
  ConvertClientConceal(const ContextAwareTypeConverter& typeConverter,
                       mlir::MLIRContext* context, bool usePublicKey,
                       polynomial::RingAttr ring)
      : ContextAwareOpConversionPattern<secret::ConcealOp>(typeConverter,
                                                           context),
        usePublicKey(usePublicKey),
        ring(ring) {}

  LogicalResult matchAndRewrite(
      secret::ConcealOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override;

 private:
  bool usePublicKey;
  polynomial::RingAttr ring;
};

// Lower a client decryption function's secret.reveal op to lwe.rlwe_decrypt +
// lwe.rlwe_decode. Modifies the containing function to add new secret key
// material args.
struct ConvertClientReveal
    : public ContextAwareOpConversionPattern<secret::RevealOp> {
  ConvertClientReveal(const ContextAwareTypeConverter& typeConverter,
                      mlir::MLIRContext* context, polynomial::RingAttr ring)
      : ContextAwareOpConversionPattern<secret::RevealOp>(typeConverter,
                                                          context),
        ring(ring) {}

  LogicalResult matchAndRewrite(
      secret::RevealOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override;

 private:
  polynomial::RingAttr ring;
};

struct ConvertExtractSlice
    : public SecretGenericOpConversion<tensor::ExtractSliceOp,
                                       tensor::ExtractOp> {
  using SecretGenericOpConversion<tensor::ExtractSliceOp,
                                  tensor::ExtractOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override;
};

struct ConvertInsertSlice
    : public SecretGenericOpConversion<tensor::InsertSliceOp,
                                       tensor::InsertOp> {
  using SecretGenericOpConversion<tensor::InsertSliceOp,
                                  tensor::InsertOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override;
};

// An empty ciphertext-semantic tensor can be used as the initializer of a
// reduction. In this case, there is no containing secret.generic op, and we
// anchor on the subsequent `mgmt::InitOp` to determine how to convert it to a
// tensor.empty whose element type is a ciphertext type.
struct ConvertEmpty : public ContextAwareOpConversionPattern<mgmt::InitOp> {
  using ContextAwareOpConversionPattern<
      mgmt::InitOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      mgmt::InitOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_PATTERNS_H_
