#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

auto &kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
auto &kOriginalTypeAttrName =
    tensor_ext::TensorExtDialect::kOriginalTypeAttrName;

#define GEN_PASS_DEF_CONVERTTOCIPHERTEXTSEMANTICS
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h.inc"

bool isPowerOfTwo(int64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

int getIndexOfOpResult(Operation *op, Value result) {
  int index = 0;
  for (auto res : op->getResults()) {
    if (res == result) return index;
    ++index;
  }
  return -1;
}

// Remove the layout attribute from the defining op of a given value. Since ops
// may have multiple results, this will not delete the attribute, but rather
// set it to nullptr and expect the rest of this pass to treat a null attribute
// as meaning the type has already been converted.
void tryRemoveLayoutAttrFromDefiningOp(Value value) {
  auto *parentOp = value.getDefiningOp();
  if (!parentOp) return;

  ArrayAttr resultLayouts = parentOp->getAttrOfType<ArrayAttr>("layout");
  int resultIndex = getIndexOfOpResult(parentOp, value);
  if (resultIndex == -1) return;

  SmallVector<Attribute> newResultLayouts(resultLayouts.begin(),
                                          resultLayouts.end());
  newResultLayouts[resultIndex] = nullptr;
  parentOp->setAttr("layout",
                    ArrayAttr::get(value.getContext(), newResultLayouts));
}

// This type converter converts types like tensor<NxMxi16> where the dimensions
// represent tensor-semantic data to tensor<ciphertext_count x num_slots x
// i16>, where the last dimension represents the ciphertext or plaintext slot
// count, and the other dimensions are determined by a layout attribute
// indexing.
//
// The presence of a layout attribute on the op definine a value is required
// for this type converter to trigger. So patterns that use this and convert
// types must remove any layout attributes when they are done.
struct LayoutMaterializationTypeConverter : public ContextAwareTypeConverter {
 public:
  LayoutMaterializationTypeConverter(int ciphertextSize)
      : ciphertextSize(ciphertextSize) {
    addConversion([&](Type type, Value v) -> std::optional<Type> {
      auto result = getContextualAttr(v);
      if (failed(result)) return std::nullopt;
      return convert(type, result.value());
    });

    addConversion([&](Type type, Value v) { return type; });
  }

  FailureOr<Type> convert(Type type, Attribute attr) const override {
    LLVM_DEBUG(llvm::dbgs() << "Converting type " << type << " with layout "
                            << attr << "\n");
    // Convert secret<tensor<...>> to secret<tensor<...>>
    // Convert tensor<...> to tensor<...>
    bool isSecret = isa<secret::SecretType>(type);
    if (isSecret) {
      auto secretType = cast<secret::SecretType>(type);
      auto innerType = secretType.getValueType();
      auto convertedInnerType = convert(innerType, attr);
      if (failed(convertedInnerType)) return failure();
      return secret::SecretType::get(convertedInnerType.value());
    }

    auto rankedTensorType = dyn_cast<RankedTensorType>(type);
    if (!rankedTensorType) return failure();

    auto layoutAttr = dyn_cast<AffineMapAttr>(attr);
    if (!layoutAttr) return failure();
    AffineMap layout = layoutAttr.getValue();

    MLIRContext *ctx = type.getContext();
    OpBuilder b(ctx);

    // Each ciphertext will always have ciphertextSize many slots, so the main
    // goal is to determine how many ciphertexts are needed. We do this by
    // iterating over the input type's index domain, and apply the layout
    // affine map to each index, and keep track of the maximum value of each
    // index of the map results. These maxima (plus 1 for zero indexing)
    // will be the shape of the new type.
    SmallVector<int64_t> outputTensorShape(layout.getNumResults(), 0);
    outputTensorShape[layout.getNumResults() - 1] = ciphertextSize;

    // Evaluate the affine map on the input indices and update the
    // outputTensorShape to be a max over visited indices.
    IndexTupleConsumer evaluateNextIndex =
        [&](const std::vector<int64_t> &indices) {
          SmallVector<Attribute> mapInputs = llvm::map_to_vector(
              indices,
              [&](int64_t i) { return cast<Attribute>(b.getIndexAttr(i)); });

          // Evaluate the affine map on the inputs
          SmallVector<Attribute> results;
          if (failed(layout.constantFold(mapInputs, results))) {
            assert(false && "constant folding should never fail here");
          }

          // minus 1 to skip the last dimension (ciphertext dimension)
          for (int i = 0; i < layout.getNumResults() - 1; ++i) {
            // 1 + to account for zero indexing
            outputTensorShape[i] =
                std::max(outputTensorShape[i],
                         1 + cast<IntegerAttr>(results[i]).getInt());
          }
        };

    iterateIndices(rankedTensorType.getShape(), evaluateNextIndex);
    return RankedTensorType::get(outputTensorShape,
                                 rankedTensorType.getElementType());
  }

  // Each value is expected to be produced by an operation whose `layout`
  // attributes correspond to the chosen layouts of the operation results.
  FailureOr<Attribute> getContextualAttr(Value value) const override {
    auto *parentOp = value.getDefiningOp();

    // It may be a block argument
    if (!parentOp) {
      // It may be a func arg
      auto blockArg = dyn_cast<BlockArgument>(value);
      auto *parentOp = blockArg.getOwner()->getParentOp();
      auto funcOp = dyn_cast<FunctionOpInterface>(parentOp);
      if (funcOp) {
        auto argAttr =
            funcOp.getArgAttr(blockArg.getArgNumber(), kLayoutAttrName);
        if (!argAttr) return failure();
        LLVM_DEBUG(llvm::dbgs()
                   << "Found layout attr " << argAttr << " on function args\n");

        return argAttr;
      }

      // It may be a secret.generic arg
      auto genericOp = dyn_cast<secret::GenericOp>(parentOp);
      if (genericOp) {
        auto attr = dyn_cast_or_null<AffineMapAttr>(
            genericOp.getArgAttr(blockArg.getArgNumber(), "layout"));
        if (!attr) return failure();
        LLVM_DEBUG(llvm::dbgs()
                   << "Found layout attr " << attr
                   << " on secret generic, for value " << value << "\n");
        return attr;
      }

      return failure();
    }

    // For any other op, the layout attribute is an array of result layouts
    ArrayAttr resultLayouts = parentOp->getAttrOfType<ArrayAttr>("layout");
    if (!resultLayouts) return failure();
    int resultIndex = getIndexOfOpResult(parentOp, value);
    if (resultIndex == -1) {
      return failure();
    }

    auto attr = dyn_cast_or_null<AffineMapAttr>(resultLayouts[resultIndex]);
    if (!attr) return failure();
    LLVM_DEBUG(llvm::dbgs() << "Found layout attr " << attr
                            << " on defining op, for value " << value << "\n");
    return attr;
  }

 private:
  // The number of slots available in each ciphertext.
  int ciphertextSize;
};

bool hasLayoutArgAttrs(func::FuncOp op) {
  for (int i = 0; i < op.getNumArguments(); ++i) {
    if (op.getArgAttr(i, kLayoutAttrName)) return true;
  }
  return false;
}

bool hasLayoutArgAttrs(secret::GenericOp op) {
  for (int i = 0; i < op.getNumOperands(); ++i) {
    if (op.getArgAttr(i, "layout")) return true;
  }
  return false;
}

bool hasLayoutResultAttrs(Operation *op) {
  auto layoutAttrs = op->getAttrOfType<ArrayAttr>("layout");
  if (!layoutAttrs) return false;

  // If any layout attribute is nullptr, it means the type has already been
  // converted. If any result has a non-null attribute, it still needs to be
  // converted.
  return llvm::any_of(layoutAttrs,
                      [](Attribute attr) { return attr != nullptr; });
}

bool hasOperandsWithLayouts(Operation *op) {
  return llvm::any_of(op->getOperands(), [](Value operand) {
    return hasLayoutResultAttrs(operand.getDefiningOp());
  });
}

struct ConvertFunc : public ConvertFuncWithContextAwareTypeConverter {
 public:
  using ConvertFuncWithContextAwareTypeConverter::
      ConvertFuncWithContextAwareTypeConverter;

  ConvertFunc(const ContextAwareTypeConverter &typeConverter,
              MLIRContext *context)
      : ConvertFuncWithContextAwareTypeConverter(typeConverter, context) {}

  LogicalResult finalizeFuncOpModification(
      func::FuncOp op, ArrayRef<Type> oldArgTypes,
      ArrayRef<Type> oldResultTypes, PatternRewriter &rewriter) const override {
    // Replace layout arg attrs with secret.original_type arg attrs This is
    // necessary so that later encoding/decoding functions can know what the
    // original type of the tensor was and how it was encoded.
    rewriter.modifyOpInPlace(op, [&] {
      for (int i = 0; i < op.getNumArguments(); ++i) {
        auto layoutAttr = op.getArgAttr(i, kLayoutAttrName);
        if (!layoutAttr) continue;

        op.removeArgAttr(i, kLayoutAttrName);
        AffineMap layout = cast<AffineMapAttr>(layoutAttr).getValue();
        op.setArgAttr(i, kOriginalTypeAttrName,
                      tensor_ext::OriginalTypeAttr::get(
                          getContext(), oldArgTypes[i], layout));
      }

      for (int i = 0; i < op.getNumResults(); ++i) {
        auto layoutAttr = dyn_cast_or_null<AffineMapAttr>(
            op.getResultAttr(i, kLayoutAttrName));
        if (!layoutAttr) continue;

        op.setResultAttr(
            i, kOriginalTypeAttrName,
            tensor_ext::OriginalTypeAttr::get(getContext(), oldResultTypes[i],
                                              layoutAttr.getValue()));
      }

      // Since the func.return was converted, we need to erase layout ops from
      // the operations that generated the return's operands.
      auto returnOperands = op.getBody().front().getTerminator()->getOperands();
      for (auto returnOperand : returnOperands) {
        tryRemoveLayoutAttrFromDefiningOp(returnOperand);
      }
    });
    return success();
  };
};

struct ConvertGeneric : public SecretGenericConversion {
  using SecretGenericConversion::SecretGenericConversion;

 public:
  LogicalResult finalizeOpModification(
      secret::GenericOp op,
      ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "Finalizing secret.generic conversion for " << op << "\n");
    rewriter.modifyOpInPlace(op, [&] {
      for (int i = 0; i < op.getNumOperands(); ++i) {
        op.removeArgAttr(i, "layout");
      }

      for (int i = 0; i < op.getNumResults(); ++i) {
        op->removeAttr("layout");
      }
    });
    LLVM_DEBUG(llvm::dbgs() << "Post-Finalization: " << op << "\n");
    return success();
  };
};

// A clone of ConvertAny<> but which erases the layout attribute afterward.
struct ConvertAnyRemovingLayout : public ConversionPattern {
  ConvertAnyRemovingLayout(const TypeConverter &anyTypeConverter,
                           MLIRContext *context)
      : ConversionPattern(anyTypeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/0, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Operation *> result =
        convertAnyOperand(getTypeConverter(), op, operands, rewriter);
    if (failed(result)) return failure();

    Operation *newOp = result.value();
    rewriter.modifyOpInPlace(newOp, [&] { newOp->removeAttr("layout"); });
    return success();
  }
};

struct ConvertToCiphertextSemantics
    : impl::ConvertToCiphertextSemanticsBase<ConvertToCiphertextSemantics> {
  using ConvertToCiphertextSemanticsBase::ConvertToCiphertextSemanticsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    LayoutMaterializationTypeConverter typeConverter =
        LayoutMaterializationTypeConverter(ciphertextSize);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return !hasLayoutArgAttrs(op); });
    target.addDynamicallyLegalOp<secret::GenericOp>(
        [&](secret::GenericOp op) { return !hasLayoutArgAttrs(op); });
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return !hasLayoutResultAttrs(op); });

    patterns.add<ConvertFunc, ConvertGeneric, ConvertAnyRemovingLayout>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
