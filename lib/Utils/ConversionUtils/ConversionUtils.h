#ifndef LIB_UTILS_CONVERSIONUTILS_CONVERSIONUTILS_H_
#define LIB_UTILS_CONVERSIONUTILS_CONVERSIONUTILS_H_

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult convertAnyOperand(const TypeConverter *typeConverter,
                                Operation *op, ArrayRef<Value> operands,
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

struct ContextAwareTypeConverter : public TypeConverter {
 public:
  // Convert types of the values in the input range, taking into account the
  // context of the values (e.g., defining ops or uses).
  // NOTE that this also converts the types of the values themselves,
  // beyond just calculate the new type
  virtual void convertValueRangeTypes(
      ValueRange values, SmallVectorImpl<Type> &newTypes) const = 0;

  // Convert types of the results of an op, taking into account the context of
  // the op when selecting the new type.
  // NOTE that this also converts the types of the results themselves,
  // beyond just calculate the new type
  virtual void convertOpResultTypes(
      Operation *op, SmallVectorImpl<Type> &newResultTypes) const = 0;

  // Convert types of the arguments and results of a function, taking into
  // account the context of the function when selecting the new types.
  // Note that this method is not used for converting the function type itself.
  // NOTE that this also converts the types of the arguments/results themselves,
  // beyond just calculate the new type
  virtual void convertFuncArgumentAndResultTypes(
      FunctionOpInterface funcOp, SmallVectorImpl<Type> &newArgTypes,
      SmallVectorImpl<Type> &newResultTypes) const = 0;
};

struct ConvertFuncWithContextAwareTypeConverter
    : public OpRewritePattern<func::FuncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  ConvertFuncWithContextAwareTypeConverter(
      const ContextAwareTypeConverter &contextAwareTypeConverter,
      MLIRContext *context)
      : OpRewritePattern(context),
        contextAwareTypeConverter(&contextAwareTypeConverter) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = cast<func::FuncOp>(op);

    SmallVector<Type> newFuncOperandsType;
    SmallVector<Type> newFuncResultsType;
    contextAwareTypeConverter->convertFuncArgumentAndResultTypes(
        op, newFuncOperandsType, newFuncResultsType);

    // update the signature
    auto newFuncType = FunctionType::get(getContext(), newFuncOperandsType,
                                         newFuncResultsType);
    rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newFuncType); });

    return success();
  }

 private:
  const ContextAwareTypeConverter *contextAwareTypeConverter;
};

template <typename T, typename Y = T>
class SecretGenericOpConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  using OpConversionPattern<secret::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (op.getBody()->getOperations().size() > 2) {
      // Each secret.generic should contain at most one instruction -
      // secret-distribute-generic can be used to distribute through the
      // arithmetic ops.
      return failure();
    }

    auto &innerOp = op.getBody()->getOperations().front();
    if (!isa<T>(innerOp)) {
      return failure();
    }

    // NOTE: use C++ RTTI instead of LLVM RTTI
    // because TypeConverter does not support LLVM RTTI
    const auto *contextAwareTypeConverter =
        dynamic_cast<const ContextAwareTypeConverter *>(getTypeConverter());

    // Assemble the arguments for the Scheme operation.
    SmallVector<Value> inputs;
    for (OpOperand &operand : innerOp.getOpOperands()) {
      if (auto *secretArg = op.getOpOperandForBlockArgument(operand.get())) {
        inputs.push_back(
            adaptor.getODSOperands(0)[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand.get());
      }
    }

    if (contextAwareTypeConverter) {
      // manually do the OpAdaptor's work
      SmallVector<Type> inputTypes;
      contextAwareTypeConverter->convertValueRangeTypes(inputs, inputTypes);
    }
    // else OpAdaptor will do it for us

    // convert the result types
    SmallVector<Type> resultTypes;
    if (contextAwareTypeConverter) {
      contextAwareTypeConverter->convertOpResultTypes(op, resultTypes);
    } else {
      auto result =
          getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes);
      if (failed(result)) return failure();
    }

    return matchAndRewriteInner(op, resultTypes, inputs, innerOp.getAttrs(),
                                rewriter);
  }

  // Default method for replacing the secret.generic with the target
  // operation.
  virtual LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs, attributes);
    return success();
  }
};

template <typename T, typename Y>
class SecretGenericOpCipherConversion : public SecretGenericOpConversion<T, Y> {
 public:
  using SecretGenericOpConversion<T, Y>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    // Check that all inputs are ciphertext.
    if (!llvm::all_of(inputs, [&](Value input) {
          return isa<lwe::RLWECiphertextType>(input.getType());
        })) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs);
    return success();
  }
};

template <typename T, typename Y>
class SecretGenericOpCipherPlainConversion
    : public SecretGenericOpConversion<T, Y> {
 public:
  using SecretGenericOpConversion<T, Y>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    auto ciphertextValues =
        llvm::to_vector(llvm::make_filter_range(inputs, [&](Value input) {
          return isa<lwe::RLWECiphertextType>(input.getType());
        }));
    if (ciphertextValues.size() != 1) {
      return failure();
    }
    auto ciphertext =
        dyn_cast<TypedValue<lwe::RLWECiphertextType>>(ciphertextValues[0]);
    if (!ciphertext) {
      return failure();
    }

    auto cleartextValues =
        llvm::to_vector(llvm::make_filter_range(inputs, [&](Value input) {
          // The cleartext value could be a tensor of values or a scalar.
          return !isa<lwe::RLWECiphertextType>(input.getType());
        }));
    if (cleartextValues.size() != 1) {
      return failure();
    }
    Value cleartext = cleartextValues[0];
    auto ciphertextTy = ciphertext.getType();
    auto plaintextTy = lwe::RLWEPlaintextType::get(
        op.getContext(), ciphertextTy.getEncoding(),
        ciphertextTy.getRlweParams().getRing(), cleartext.getType());
    auto plaintext = rewriter.create<lwe::RLWEEncodeOp>(
        op.getLoc(), plaintextTy, cleartext, ciphertextTy.getEncoding(),
        ciphertextTy.getRlweParams().getRing());

    rewriter.replaceOpWithNewOp<Y>(op, ciphertext, plaintext);
    return success();
  }
};

template <typename T>
class SecretGenericOpRelinearizeConversion
    : public SecretGenericOpConversion<mgmt::RelinearizeOp, T> {
 public:
  using SecretGenericOpConversion<mgmt::RelinearizeOp,
                                  T>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    auto inputDimension = cast<lwe::RLWECiphertextType>(inputs[0].getType())
                              .getRlweParams()
                              .getDimension();
    SmallVector<int32_t> fromBasis;
    for (int i = 0; i < inputDimension; ++i) {
      fromBasis.push_back(i);
    }
    SmallVector<int32_t> toBasis = {0, 1};

    rewriter.replaceOpWithNewOp<T>(op, inputs[0],
                                   rewriter.getDenseI32ArrayAttr(fromBasis),
                                   rewriter.getDenseI32ArrayAttr(toBasis));
    return success();
  }
};

template <typename M, typename T, typename Y>
class SecretGenericOpMulConversion : public SecretGenericOpConversion<M, T> {
 public:
  using SecretGenericOpConversion<M, T>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    auto plaintextValues =
        llvm::to_vector(llvm::make_filter_range(inputs, [&](Value input) {
          return !isa<lwe::RLWECiphertextType>(input.getType());
        }));
    if (!plaintextValues.empty()) {
      return failure();
    }

    // only left for CKKS, should be removed later
    rewriter.replaceOpWithNewOp<Y>(op, rewriter.create<T>(op.getLoc(), inputs),
                                   rewriter.getDenseI32ArrayAttr({0, 1, 2}),
                                   rewriter.getDenseI32ArrayAttr({0, 1}));
    return success();
  }
};

template <typename T>
class SecretGenericOpRotateConversion
    : public SecretGenericOpConversion<tensor_ext::RotateOp, T> {
 public:
  using SecretGenericOpConversion<tensor_ext::RotateOp,
                                  T>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    // Check that the offset is a constant.
    auto offset = inputs[1];
    auto constantOffset =
        dyn_cast<mlir::arith::ConstantOp>(offset.getDefiningOp());
    if (!constantOffset) {
      op.emitError("expected constant offset for rotate");
    }
    auto offsetAttr = llvm::dyn_cast<IntegerAttr>(constantOffset.getValue());

    rewriter.replaceOpWithNewOp<T>(op, inputs[0], offsetAttr);
    return success();
  }
};

template <typename Y>
class SecretGenericOpModulusSwitchConversion
    : public SecretGenericOpConversion<mgmt::ModReduceOp, Y> {
 public:
  using SecretGenericOpConversion<mgmt::ModReduceOp,
                                  Y>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    auto outputType = outputTypes[0];
    auto outputRing =
        cast<lwe::RLWECiphertextType>(outputType).getRlweParams().getRing();

    rewriter.replaceOpWithNewOp<Y>(op, outputTypes[0], inputs[0], outputRing);
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
  for (auto block_arg : op->getParentOfType<func::FuncOp>()
                            .getBody()
                            .getBlocks()
                            .front()
                            .getArguments()) {
    if (mlir::isa<ArgType>(block_arg.getType())) {
      return block_arg;
    }
  }
  return failure();
}

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

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CONVERSIONUTILS_CONVERSIONUTILS_H_
