#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_H_

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DECL
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h.inc"

class ToOpenfheTypeConverter : public TypeConverter {
 public:
  ToOpenfheTypeConverter(MLIRContext* ctx);
};

FailureOr<Value> getContextualCryptoContext(Operation* op);

template <typename UnaryOp, typename OpenfheOp>
struct ConvertUnaryOp : public OpConversionPattern<UnaryOp> {
  using OpConversionPattern<UnaryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnaryOp op, typename UnaryOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op,
                       OpenfheOp::create(rewriter, op.getLoc(), cryptoContext,
                                         adaptor.getInput()));
    return success();
  }
};

template <typename BinOp, typename OpenfheOp>
struct ConvertLWEBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<OpenfheOp>(
        op, openfhe::CiphertextType::get(op.getContext()), cryptoContext,
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

template <typename BinOp, typename OpenfheOp>
struct ConvertCiphertextPlaintextOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<OpenfheOp>(
        op, openfhe::CiphertextType::get(op.getContext()), cryptoContext,
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

template <typename RotateOp, typename OpenfheOp>
struct ConvertRotateOp : public OpConversionPattern<RotateOp> {
  ConvertRotateOp(mlir::MLIRContext* context)
      : OpConversionPattern<RotateOp>(context) {}

  using OpConversionPattern<RotateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RotateOp op, typename RotateOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(
        op, OpenfheOp::create(rewriter, op.getLoc(), cryptoContext,
                              adaptor.getInput(), adaptor.getOffset()));
    return success();
  }
};

inline bool checkRelinToBasis(llvm::ArrayRef<int> toBasis) {
  if (toBasis.size() != 2) return false;
  return toBasis[0] == 0 && toBasis[1] == 1;
}

template <typename RelinOp, typename OpenfheOp>
struct ConvertRelinOp : public OpConversionPattern<RelinOp> {
  ConvertRelinOp(mlir::MLIRContext* context)
      : OpConversionPattern<RelinOp>(context) {}

  using OpConversionPattern<RelinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RelinOp op, typename RelinOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;
    Value cryptoContext = result.value();

    auto toBasis = adaptor.getToBasis();

    // Since the `Relinearize()` function in OpenFHE relinearizes a ciphertext
    // to the lowest level (for (1,s)), the `to_basis` of `<scheme>.RelinOp`
    // must be [0,1].
    if (!checkRelinToBasis(toBasis)) {
      op.emitError() << "toBasis must be [0, 1], got [" << toBasis << "]";
      return failure();
    }
    rewriter.replaceOpWithNewOp<OpenfheOp>(
        op, openfhe::CiphertextType::get(op.getContext()), cryptoContext,
        adaptor.getInput());
    return success();
  }
};

// for CKKS, it is called Rescale but internally for OpenFHE it is an
// alias for openfhe::ModReduceOp
template <typename ModulusSwitchOp>
struct ConvertModulusSwitchOp : public OpConversionPattern<ModulusSwitchOp> {
  ConvertModulusSwitchOp(mlir::MLIRContext* context)
      : OpConversionPattern<ModulusSwitchOp>(context) {}

  using OpConversionPattern<ModulusSwitchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModulusSwitchOp op, typename ModulusSwitchOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, openfhe::ModReduceOp::create(
                               rewriter, op.getLoc(),
                               openfhe::CiphertextType::get(op.getContext()),
                               cryptoContext, adaptor.getInput()));
    return success();
  }
};

template <typename LevelReduceOp>
struct ConvertLevelReduceOp : public OpConversionPattern<LevelReduceOp> {
  ConvertLevelReduceOp(mlir::MLIRContext* context)
      : OpConversionPattern<LevelReduceOp>(context) {}

  using OpConversionPattern<LevelReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LevelReduceOp op, typename LevelReduceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(
        op, openfhe::LevelReduceOp::create(
                rewriter, op.getLoc(),
                openfhe::CiphertextType::get(op.getContext()), cryptoContext,
                adaptor.getInput(), op.getLevelToDrop()));
    return success();
  }
};

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_H_
