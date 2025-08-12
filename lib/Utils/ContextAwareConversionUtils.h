#ifndef LIB_UTILS_CONTEXTAWARECOVNVERSIONUTILS_H_
#define LIB_UTILS_CONTEXTAWARECOVNVERSIONUTILS_H_

#include <cstdint>
#include <functional>
#include <numeric>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETraits.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Replace the input op with a new op where all operands and results have been
// replaced by their type-converted versions, and all regions. Note that the
// input ArrayRef<Value> for operands must already be type converted (this is
// true when this is called from a ContextAwareConversionPattern).
FailureOr<Operation*> convertGeneral(
    const ContextAwareTypeConverter* typeConverter, Operation* op,
    ArrayRef<Value> operands, ContextAwareConversionPatternRewriter& rewriter);

template <typename T = void>
struct ConvertAnyContextAware : public ContextAwareConversionPattern {
  ConvertAnyContextAware(
      const ContextAwareTypeConverter& anyContextAwareTypeConverter,
      MLIRContext* context, int benefit = 1)
      : ContextAwareConversionPattern(anyContextAwareTypeConverter,
                                      RewritePattern::MatchAnyOpTypeTag(),
                                      benefit, context) {
    setDebugName("ConvertAnyContextAware");
    setHasBoundedRewriteRecursion(true);
  }

  // A hook to allow postprocessing of the op after it is otherwise generically
  // converted. This is useful in context-aware dialect conversion because the
  // context is often an attribute attached to an op, and we may want to remove
  // that attribute as part of the means to signal the conversion is done.
  virtual LogicalResult finalizeOpModification(
      T op, ContextAwareConversionPatternRewriter& rewriter) const {
    return success();
  };

  // Generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions, and all regions have their block
  // signatures converter. Note the input ArrayRef<Value> is already type
  // converted.
  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    if (!isa<T>(op)) {
      return failure();
    }

    auto result = convertGeneral(getTypeConverter(), op, operands, rewriter);
    if (failed(result)) return failure();

    return finalizeOpModification(cast<T>(result.value()), rewriter);
  }
};

template <>
struct ConvertAnyContextAware<void> : public ContextAwareConversionPattern {
  ConvertAnyContextAware<void>(
      const ContextAwareTypeConverter& anyContextAwareTypeConverter,
      MLIRContext* context, int benefit = 1)
      : ContextAwareConversionPattern(anyContextAwareTypeConverter,
                                      RewritePattern::MatchAnyOpTypeTag(),
                                      benefit, context) {
    setDebugName("ConvertAnyContextAware");
    setHasBoundedRewriteRecursion(true);
  }

  // A hook to allow postprocessing of the op after it is otherwise generically
  // converted. This is useful in context-aware dialect conversion because the
  // context is often an attribute attached to an op, and we may want to remove
  // that attribute as part of the means to signal the conversion is done.
  virtual LogicalResult finalizeOpModification(
      Operation* op, ContextAwareConversionPatternRewriter& rewriter) const {
    return success();
  };

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto result = convertGeneral(getTypeConverter(), op, operands, rewriter);
    if (failed(result)) return failure();

    return finalizeOpModification(result.value(), rewriter);
  }
};

template <typename T, typename Y = T>
class SecretGenericOpConversion
    : public ContextAwareOpConversionPattern<secret::GenericOp> {
 public:
  using ContextAwareOpConversionPattern<
      secret::GenericOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
    if (op.getBody()->getOperations().size() > 2) {
      // Each secret.generic should contain at most one instruction -
      // secret-distribute-generic can be used to distribute through the
      // arithmetic ops.
      return failure();
    }

    auto& innerOp = op.getBody()->getOperations().front();
    if (!isa<T>(innerOp)) {
      return failure();
    }

    // The inner op's arguments are either plaintext operands, in which case
    // they are already type-converted, or else they are ciphertext operands,
    // in which case we can get them in type-converted form from the adaptor.
    SmallVector<Value> inputs;
    for (Value operand : innerOp.getOperands()) {
      if (auto* secretArg = op.getOpOperandForBlockArgument(operand)) {
        inputs.push_back(adaptor.getInputs()[secretArg->getOperandNumber()]);
      } else {
        inputs.push_back(operand);
      }
    }

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                op.getResults(), resultTypes)))
      return failure();

    // only preserve dialect attrs
    // we do not want op attrs like overflowFlags from arith.add
    SmallVector<NamedAttribute> attrsToPreserve;
    for (auto& namedAttr : innerOp.getDialectAttrs()) {
      attrsToPreserve.push_back(namedAttr);
    }

    // Must preserve mgmt attrs from the enclosing generic so that later ops
    // have access to it for type conversion. However, the generic op uses
    // OperandAndResultAttrInterface, and this gives special names to the
    // attributes, so it doesn't make sense to copy those names to the
    // converted op.
    convertArrayOfDicts(op.getAllOperandAttrsAttr(), attrsToPreserve);
    convertArrayOfDicts(op.getAllResultAttrsAttr(), attrsToPreserve);
    DenseSet<StringRef> seenNames;
    SmallVector<NamedAttribute> dedupedAttrsToPreserve;
    for (NamedAttribute preservedAttr : attrsToPreserve) {
      if (!seenNames.insert(preservedAttr.getName()).second) {
        continue;  // Skip duplicates.
      }
      dedupedAttrsToPreserve.push_back(preservedAttr);
    }

    FailureOr<Operation*> newOpResult = matchAndRewriteInner(
        op, resultTypes, inputs, dedupedAttrsToPreserve, rewriter);
    if (failed(newOpResult)) return failure();
    Operation* newOp = newOpResult.value();

    // The subclass may intentionally set some attribute that we would have
    // otherwise preserved. If this is the case, don't set that attribute in
    // the new op.
    DictionaryAttr existingAttrs(newOp->getAttrDictionary());
    SmallVector<NamedAttribute> attrsToSet(existingAttrs.getValue());
    for (NamedAttribute preservedAttr : dedupedAttrsToPreserve) {
      if (existingAttrs.get(preservedAttr.getName())) {
        continue;
      }
      attrsToSet.push_back(preservedAttr);
    }
    rewriter.modifyOpInPlace(newOp, [&]() { newOp->setAttrs(attrsToSet); });
    return success();
  }

  // Default method for replacing the secret.generic with the target
  // operation.
  virtual FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const {
    return rewriter.replaceOpWithNewOp<Y>(op, outputTypes, inputs, attributes)
        .getOperation();
  }
};

template <typename T, typename Y>
class SecretGenericOpCipherPlainConversion
    : public SecretGenericOpConversion<T, Y> {
 public:
  using SecretGenericOpConversion<T, Y>::SecretGenericOpConversion;

  // Ciphertext-plaintext ops should take precedence over ciphertext-ciphertext
  // ops because the ops being converted (e.g., addi) don't have a plaintext
  // variant.
  SecretGenericOpCipherPlainConversion(
      const ContextAwareTypeConverter& typeConverter, MLIRContext* context)
      : SecretGenericOpConversion<T, Y>(typeConverter, context, /*benefit=*/2) {
  }

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    // Verify that exactly one of the two inputs is a ciphertext.
    if (inputs.size() != 2 ||
        llvm::count_if(inputs, [&](Value input) {
          return isa<lwe::LWECiphertextType>(input.getType());
        }) != 1) {
      return failure();
    }

    auto encodeCleartext =
        [&](Value cleartext,
            lwe::LWECiphertextType ciphertextTy) -> FailureOr<Value> {
      auto initOp = dyn_cast_or_null<mgmt::InitOp>(cleartext.getDefiningOp());
      if (!initOp) {
        return failure();
      }
      auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(initOp);
      if (!mgmtAttr) {
        return failure();
      }

      Attribute ciphertextEncoding =
          ciphertextTy.getPlaintextSpace().getEncoding();
      Attribute plaintextEncoding = lwe::getEncodingAttrWithNewScalingFactor(
          ciphertextEncoding, mgmtAttr.getScale());

      if (!plaintextEncoding) {
        return failure();
      }

      // TODO(#1643): inherit level information to plaintext type from init-op
      // mgmt attr. This actually needs to make LWEPlaintextType RNS aware.
      auto plaintextTy = lwe::LWEPlaintextType::get(
          op.getContext(), ciphertextTy.getApplicationData(),
          lwe::PlaintextSpaceAttr::get(
              op.getContext(), ciphertextTy.getPlaintextSpace().getRing(),
              plaintextEncoding));
      Value realCleartext = initOp.getInput();
      return rewriter
          .create<lwe::RLWEEncodeOp>(op.getLoc(), plaintextTy, realCleartext,
                                     plaintextEncoding,
                                     ciphertextTy.getPlaintextSpace().getRing())
          .getResult();
    };

    Value input0 = inputs[0];
    Value input1 = inputs[1];
    if (auto ciphertextTy =
            dyn_cast<lwe::LWECiphertextType>(input0.getType())) {
      auto encoded = encodeCleartext(input1, ciphertextTy);
      if (failed(encoded)) {
        return failure();
      }
      auto newOp = rewriter.replaceOpWithNewOp<Y>(op, input0, *encoded);
      return newOp.getOperation();
    }

    auto encoded =
        encodeCleartext(input0, cast<lwe::LWECiphertextType>(input1.getType()));
    if (failed(encoded)) {
      return failure();
    }
    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, *encoded, input1);
    return newOp.getOperation();
  }
};

template <typename T>
class SecretGenericOpRelinearizeConversion
    : public SecretGenericOpConversion<mgmt::RelinearizeOp, T> {
 public:
  using SecretGenericOpConversion<mgmt::RelinearizeOp,
                                  T>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto inputDimension = cast<lwe::LWECiphertextType>(inputs[0].getType())
                              .getCiphertextSpace()
                              .getSize();
    SmallVector<int32_t> fromBasis;
    for (int i = 0; i < inputDimension; ++i) {
      fromBasis.push_back(i);
    }
    SmallVector<int32_t> toBasis = {0, 1};

    auto newOp = rewriter.replaceOpWithNewOp<T>(
        op, inputs[0], rewriter.getDenseI32ArrayAttr(fromBasis),
        rewriter.getDenseI32ArrayAttr(toBasis));
    return newOp.getOperation();
  }
};

template <typename T>
class SecretGenericOpRotateConversion
    : public SecretGenericOpConversion<tensor_ext::RotateOp, T> {
 public:
  using SecretGenericOpConversion<tensor_ext::RotateOp,
                                  T>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    // Check that the offset is a constant.
    auto offset = inputs[1];
    auto constantOffset =
        dyn_cast<mlir::arith::ConstantOp>(offset.getDefiningOp());
    if (!constantOffset) {
      op.emitError("expected constant offset for rotate");
    }
    auto offsetAttr = llvm::dyn_cast<IntegerAttr>(constantOffset.getValue());
    auto newOp =
        rewriter.replaceOpWithNewOp<T>(op, outputTypes, inputs[0], offsetAttr);
    return newOp.getOperation();
  }
};

template <typename Y>
class SecretGenericOpModulusSwitchConversion
    : public SecretGenericOpConversion<mgmt::ModReduceOp, Y> {
 public:
  using SecretGenericOpConversion<mgmt::ModReduceOp,
                                  Y>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto outputType = outputTypes[0];
    auto outputElementType = getElementTypeOrSelf(outputType);
    auto outputRing = cast<lwe::LWECiphertextType>(outputElementType)
                          .getCiphertextSpace()
                          .getRing();

    // secret-to-ckks allow tensor of ciphertext
    if (auto outputTensorType = dyn_cast<RankedTensorType>(outputType)) {
      // need tensor::extract/tensor::insert
      // manually fully unroll it
      auto shape = outputTensorType.getShape();
      auto totalSize = std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<int64_t>());
      auto emptyOp = rewriter.create<tensor::EmptyOp>(op.getLoc(), shape,
                                                      outputElementType);
      Operation* resultOp = emptyOp;
      for (int i = 0; i < totalSize; ++i) {
        SmallVector<int64_t> indices;
        auto iCopy = i;
        for (int64_t j : shape) {
          indices.push_back(iCopy % j);
          iCopy /= j;
        }
        SmallVector<Value> constants;
        for (int64_t index : indices) {
          constants.push_back(rewriter.create<mlir::arith::ConstantOp>(
              op.getLoc(), rewriter.getIndexAttr(index)));
        }
        auto extract = rewriter.create<tensor::ExtractOp>(op.getLoc(),
                                                          inputs[0], constants);
        auto modulusSwitchOp = rewriter.create<Y>(
            op.getLoc(), outputElementType, extract.getResult(), outputRing);
        modulusSwitchOp->setDialectAttrs(attributes);
        auto insert = rewriter.create<tensor::InsertOp>(
            op.getLoc(), modulusSwitchOp.getResult(), resultOp->getResult(0),
            constants);
        resultOp = insert;
      }
      rewriter.replaceOp(op, resultOp);
      return resultOp;
    }

    auto newOp = rewriter.replaceOpWithNewOp<Y>(op, outputTypes[0], inputs[0],
                                                outputRing);
    return newOp.getOperation();
  }
};

template <typename T>
class SecretGenericOpLevelReduceConversion
    : public SecretGenericOpConversion<mgmt::LevelReduceOp, T> {
 public:
  using SecretGenericOpConversion<mgmt::LevelReduceOp,
                                  T>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto innerOp =
        cast<mgmt::LevelReduceOp>(op.getBody()->getOperations().front());
    auto levelToDrop = innerOp.getLevelToDrop();

    auto newOp =
        rewriter.replaceOpWithNewOp<T>(op, outputTypes, inputs[0], levelToDrop);
    return newOp.getOperation();
  }
};

struct ContextAwareFuncConversion
    : public ContextAwareOpConversionPattern<func::FuncOp> {
 public:
  using ContextAwareOpConversionPattern::ContextAwareOpConversionPattern;

  ContextAwareFuncConversion(
      const ContextAwareTypeConverter& contextAwareTypeConverter,
      MLIRContext* context)
      : ContextAwareOpConversionPattern(context, /*benefit=*/2),
        contextAwareTypeConverter(&contextAwareTypeConverter) {}

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const override;

  // An overridable hook that allows subclasses to perform additional
  // modifications of the func op after its type signature has been converted.
  // For example, a subclass may use this hook to modify arg attrs.
  virtual LogicalResult finalizeFuncOpModification(
      func::FuncOp op, ArrayRef<Type> oldArgTypes,
      ArrayRef<Type> oldResultTypes, PatternRewriter& rewriter) const {
    return success();
  };

 private:
  const ContextAwareTypeConverter* contextAwareTypeConverter;
};

class SecretGenericFuncCallConversion
    : public SecretGenericOpConversion<func::CallOp, func::CallOp> {
 public:
  using SecretGenericOpConversion<func::CallOp,
                                  func::CallOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override;
};

void addStructuralConversionPatterns(ContextAwareTypeConverter& typeConverter,
                                     RewritePatternSet& patterns,
                                     ConversionTarget& target);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CONTEXTAWARECOVNVERSIONUTILS_H_
