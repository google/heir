#ifndef LIB_UTILS_CONVERSIONUTILS_H_
#define LIB_UTILS_CONVERSIONUTILS_H_

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

FailureOr<Operation *> convertAnyOperand(const TypeConverter *typeConverter,
                                         Operation *op,
                                         ArrayRef<Value> operands,
                                         ConversionPatternRewriter &rewriter);

template <typename T = void>
struct ConvertAny : public ConversionPattern {
  ConvertAny(const TypeConverter &anyTypeConverter, MLIRContext *context)
      : ConversionPattern(anyTypeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<T>(op)) {
      return failure();
    }

    return convertAnyOperand(getTypeConverter(), op, operands, rewriter);
  }
};

template <>
struct ConvertAny<void> : public ConversionPattern {
  ConvertAny<void>(const TypeConverter &anyTypeConverter, MLIRContext *context)
      : ConversionPattern(anyTypeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    return convertAnyOperand(getTypeConverter(), op, operands, rewriter);
  }
};

template <typename SourceOpTy, typename TargetOpTy>
struct ConvertBinOp : public OpConversionPattern<SourceOpTy> {
  ConvertBinOp(mlir::MLIRContext *context)
      : OpConversionPattern<SourceOpTy>(context) {}

  using OpConversionPattern<SourceOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOpTy op, typename SourceOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<TargetOpTy>(adaptor.getLhs().getType(),
                                       adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename T = void>
struct DropOp : public ConversionPattern {
  DropOp(const TypeConverter &typeConverter, MLIRContext *context,
         PatternBenefit benefit = 2)
      : ConversionPattern(typeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/2, context) {
    setDebugName("DropOp");
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<T>(op)) {
      return failure();
    }

    if (op->getNumOperands() != op->getNumResults()) {
      return op->emitError()
             << "invalid use of DropOp with op having "
                "non-matching operand and result sizes; numOperands="
             << op->getNumOperands() << ", numResults=" << op->getNumResults();
    }

    rewriter.replaceOp(op, operands);
    return success();
  }
};

// Adds conversion patterns that deal with tensor<..xsource_type>
// when source_type will be type converted to tensor<...>, too
void addTensorOfTensorConversionPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         ConversionTarget &target);

// Adds the standard set of conversion patterns for
// converting types involved in func, cf, etc., which
// don't depend on the logic of the dialect beyond the
// type converter.
void addStructuralConversionPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target);

// Seems like this would be better as a method on the
// LWE_EncodingAttrWithScalingFactor class, but I still have the problem of
// the type returned by getEncoding being a vanilla Attribute. Probably we
// need a common interface for LWE_EncodingAttrWithScalingFactor, and cast to
// that?
int widthFromEncodingAttr(Attribute encoding);

// Returns the Value corresponding to a given type in the FuncOp containing
// this op.
template <typename ArgType>
FailureOr<Value> getContextualArgFromFunc(Operation *op) {
  for (auto blockArg : op->getParentOfType<func::FuncOp>()
                           .getBody()
                           .getBlocks()
                           .front()
                           .getArguments()) {
    if (mlir::isa<ArgType>(blockArg.getType())) {
      return blockArg;
    }
  }
  return failure();
}

// Returns the Value corresponding to a given type in the FuncOp containing
// this op.
FailureOr<Value> getContextualArgFromFunc(Operation *op, Type argType);

// FIXME: update this after #1196
// Returns true if the func contains ops from the given dialects.
template <typename Dialect>
bool containsLweOrDialect(func::FuncOp func) {
  auto walkResult = func.walk([&](Operation *op) {
    if (llvm::isa<Dialect, lwe::LWEDialect>(op->getDialect()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

inline Type encrytpedUIntTypeFromWidth(MLIRContext *ctx, int width) {
  // Only supporting unsigned types because the LWE dialect does not have a
  // notion of signedness.
  switch (width) {
    case 1:
      return tfhe_rust::EncryptedBoolType::get(ctx);
    case 2:
      return tfhe_rust::EncryptedUInt2Type::get(ctx);
    case 3:
      return tfhe_rust::EncryptedUInt3Type::get(ctx);
    case 4:
      return tfhe_rust::EncryptedUInt4Type::get(ctx);
    case 8:
      return tfhe_rust::EncryptedUInt8Type::get(ctx);
    case 10:
      return tfhe_rust::EncryptedUInt10Type::get(ctx);
    case 12:
      return tfhe_rust::EncryptedUInt12Type::get(ctx);
    case 14:
      return tfhe_rust::EncryptedUInt14Type::get(ctx);
    case 16:
      return tfhe_rust::EncryptedUInt16Type::get(ctx);
    case 32:
      return tfhe_rust::EncryptedUInt32Type::get(ctx);
    case 64:
      return tfhe_rust::EncryptedUInt64Type::get(ctx);
    case 128:
      return tfhe_rust::EncryptedUInt128Type::get(ctx);
    case 256:
      return tfhe_rust::EncryptedUInt256Type::get(ctx);
    default:
      llvm_unreachable("Unsupported bitwidth");
  }
}

inline Type encrytpedIntTypeFromWidth(MLIRContext *ctx, int width) {
  // Only supporting unsigned types because the LWE dialect does not have a
  // notion of signedness.
  switch (width) {
    case 1:
      return tfhe_rust::EncryptedBoolType::get(ctx);
    case 2:
      return tfhe_rust::EncryptedInt2Type::get(ctx);
    case 4:
      return tfhe_rust::EncryptedInt4Type::get(ctx);
    case 8:
      return tfhe_rust::EncryptedInt8Type::get(ctx);
    case 16:
      return tfhe_rust::EncryptedInt16Type::get(ctx);
    case 32:
      return tfhe_rust::EncryptedInt32Type::get(ctx);
    case 64:
      return tfhe_rust::EncryptedInt64Type::get(ctx);
    case 128:
      return tfhe_rust::EncryptedInt128Type::get(ctx);
    case 256:
      return tfhe_rust::EncryptedInt256Type::get(ctx);
    default:
      llvm_unreachable("Unsupported bitwidth");
  }
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CONVERSIONUTILS_H_
