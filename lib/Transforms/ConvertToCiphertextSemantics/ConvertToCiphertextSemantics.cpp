#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
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

#define GEN_PASS_DEF_CONVERTTOCIPHERTEXTSEMANTICS
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h.inc"

bool isPowerOfTwo(int64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

// This type converter converts types like tensor<NxMxi16> where the dimensions
// represent tensor-semantic data to tensor<ciphertext_count x num_slots x
// i16>, where the last dimension represents the ciphertext or plaintext slot
// count, and the other dimensions are determined by a layout attribute
// indexing.
struct LayoutMaterializationTypeConverter : public AttributeAwareTypeConverter {
 public:
  LayoutMaterializationTypeConverter(int ciphertextSize)
      : ciphertextSize(ciphertextSize) {}

  FailureOr<Type> convert(Type type, Attribute attr) const override {
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
            funcOp.getArgAttr(blockArg.getArgNumber(),
                              tensor_ext::TensorExtDialect::kLayoutAttrName);
        if (!argAttr) return failure();

        return argAttr;
      }

      // It may be a secret.generic arg
      auto genericOp = dyn_cast<secret::GenericOp>(parentOp);
      if (genericOp) {
        return cast<AffineMapAttr>(
            genericOp.getArgAttr(blockArg.getArgNumber(), "layout"));
      }

      return failure();
    }

    // For any other op, the layout attribute is an array of result layouts
    ArrayAttr resultLayouts = parentOp->getAttrOfType<ArrayAttr>("layout");

    int valueIndex = -1;
    for (auto result : parentOp->getResults()) {
      ++valueIndex;
      if (result == value) break;
    }

    if (valueIndex == -1) {
      return failure();
    }

    return cast<AffineMapAttr>(resultLayouts[valueIndex]);
  }

 private:
  // The number of slots available in each ciphertext.
  int ciphertextSize;
};

bool hasLayoutArgAttrs(func::FuncOp op) {
  for (int i = 0; i < op.getNumArguments(); ++i) {
    if (op.getArgAttr(i, tensor_ext::TensorExtDialect::kLayoutAttrName))
      return true;
  }
  return false;
}

bool hasLayoutResultAttrs(Operation *op) {
  return (op->getAttrOfType<ArrayAttr>("layout") != nullptr);
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
    // Replace layout arg attrs with secret.original_type arg attrs
    rewriter.modifyOpInPlace(op, [&] {
      for (int i = 0; i < op.getNumArguments(); ++i) {
        auto layoutAttr =
            op.getArgAttr(i, tensor_ext::TensorExtDialect::kLayoutAttrName);
        if (!layoutAttr) continue;

        op.removeArgAttr(i, tensor_ext::TensorExtDialect::kLayoutAttrName);
        AffineMap layout = cast<AffineMapAttr>(layoutAttr).getValue();
        op.setArgAttr(i, tensor_ext::TensorExtDialect::kOriginalTypeAttrName,
                      tensor_ext::OriginalTypeAttr::get(
                          getContext(), oldArgTypes[i], layout));
      }
    });
    return success();
  };
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
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return !hasLayoutResultAttrs(op); });

    patterns.add<ConvertFunc>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
