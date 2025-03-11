#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

namespace {
using tensor_ext::LayoutAttr;

auto &kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
auto &kMaterializedAttrName = "tensor_ext.layout_materialized";
auto &kOriginalTypeAttrName =
    tensor_ext::TensorExtDialect::kOriginalTypeAttrName;

}  // namespace

// An unset value of a permutation as it's being built up.
static constexpr int kUnset = -1;

#define GEN_PASS_DEF_CONVERTTOCIPHERTEXTSEMANTICS
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h.inc"

bool isPowerOfTwo(int64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

bool containsDim(ArrayRef<int64_t> dims, int64_t dim) {
  return llvm::any_of(dims, [dim](int64_t d) { return d == dim; });
}

// This type converter converts types like tensor<NxMxi16> where the dimensions
// represent tensor-semantic data to tensor<ciphertext_count x num_slots x
// i16>, where the last dimension represents the ciphertext or plaintext slot
// count, and the other dimensions are determined by a layout attribute
// indexing.
//
// The presence of a layout attribute on the op define a value is required
// for this type converter to trigger. So patterns that use this and convert
// types must remove any layout attributes when they are done.
//
// TODO(#1450): Determine if we should support non-cyclic slot algebras here
// i.e., for the usual 2xN/2 case, how would we determine this situation
// at this stage of the compilation pipeline, and how would this pass update
// to convert to tensor<AxBx2xN/2xi16> where the last two dimensions now
// correspond to the slot algebra direct product?
struct LayoutMaterializationTypeConverter
    : public UniquelyNamedAttributeAwareTypeConverter {
 public:
  LayoutMaterializationTypeConverter(int ciphertextSize)
      : UniquelyNamedAttributeAwareTypeConverter(kLayoutAttrName),
        ciphertextSize(ciphertextSize) {
    addConversion([&](Type type, Attribute attr) { return std::nullopt; });
    addConversion(
        [&](secret::SecretType type, LayoutAttr attr) -> std::optional<Type> {
          auto innerType = type.getValueType();
          auto rankedTensorType = dyn_cast<RankedTensorType>(innerType);
          if (!rankedTensorType) return std::nullopt;

          auto convertedInnerType = materializeLayout(rankedTensorType, attr);
          if (failed(convertedInnerType)) return std::nullopt;

          return secret::SecretType::get(convertedInnerType.value());
        });
    addConversion(
        [&](RankedTensorType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeLayout(type, attr);
        });
  }

  FailureOr<Type> materializeLayout(RankedTensorType type,
                                    LayoutAttr attr) const {
    AffineMap layout = attr.getMap();
    MLIRContext *ctx = type.getContext();
    OpBuilder b(ctx);
    LLVM_DEBUG(llvm::dbgs() << "Unaligned type: " << type << "\n");

    // First extract the tensor type as expanded according to the
    // alignment attribute.
    tensor_ext::AlignmentAttr alignment = attr.getAlignment();
    if (alignment) {
      type = RankedTensorType::get(alignment.getOut(), type.getElementType());
    }
    LLVM_DEBUG(llvm::dbgs() << "Aligned type: " << type << "\n");

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
          SmallVector<int64_t> results;
          evaluateStatic(layout, indices, results);

          // minus 1 to skip the last dimension (ciphertext dimension)
          for (int i = 0; i < layout.getNumResults() - 1; ++i) {
            // 1 + to account for zero indexing
            outputTensorShape[i] =
                std::max(outputTensorShape[i], 1 + results[i]);
          }
        };

    iterateIndices(type.getShape(), evaluateNextIndex);
    return RankedTensorType::get(outputTensorShape, type.getElementType());
  }

 private:
  // The number of slots available in each ciphertext.
  int ciphertextSize;
};

bool hasMaterializedAttr(Operation *op) {
  return op->hasAttr(kMaterializedAttrName);
}

void setMaterializedAttr(Operation *op) {
  op->setAttr(kMaterializedAttrName, UnitAttr::get(op->getContext()));
}

struct ConvertFunc : public ContextAwareFuncConversion {
 public:
  ConvertFunc(const ContextAwareTypeConverter &converter, MLIRContext *context)
      : ContextAwareFuncConversion(converter, context) {}

  LogicalResult finalizeFuncOpModification(
      func::FuncOp op, ArrayRef<Type> oldArgTypes,
      ArrayRef<Type> oldResultTypes, PatternRewriter &rewriter) const override {
    // Replace layout arg attrs with secret.original_type arg attrs This is
    // necessary so that later encoding/decoding functions can know what the
    // original type of the tensor was and how it was encoded.
    rewriter.modifyOpInPlace(op, [&] {
      setMaterializedAttr(op);
      for (int i = 0; i < op.getNumArguments(); ++i) {
        auto layoutAttr =
            dyn_cast_or_null<LayoutAttr>(op.getArgAttr(i, kLayoutAttrName));
        if (!layoutAttr) continue;

        op.setArgAttr(i, kOriginalTypeAttrName,
                      tensor_ext::OriginalTypeAttr::get(
                          getContext(), oldArgTypes[i], layoutAttr));
      }

      for (int i = 0; i < op.getNumResults(); ++i) {
        auto layoutAttr =
            dyn_cast_or_null<LayoutAttr>(op.getResultAttr(i, kLayoutAttrName));
        if (!layoutAttr) continue;

        op.setResultAttr(i, kOriginalTypeAttrName,
                         tensor_ext::OriginalTypeAttr::get(
                             getContext(), oldResultTypes[i], layoutAttr));
      }
    });
    return success();
  };
};

struct ConvertGeneric : public ConvertAnyContextAware<secret::GenericOp> {
 public:
  ConvertGeneric(const ContextAwareTypeConverter &converter,
                 MLIRContext *context)
      : ConvertAnyContextAware(converter, context) {
    setDebugName("ConvertGeneric");
  }

  LogicalResult finalizeOpModification(
      secret::GenericOp op,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] { setMaterializedAttr(op); });
    return success();
  };
};

// Convert an op generically, marking it as materialized. Lowest priority
// because it is only meant to handle ops that don't have special
// materialization rules.
struct ConvertAnyAddingMaterializedAttr : public ConvertAnyContextAware<> {
  ConvertAnyAddingMaterializedAttr(const ContextAwareTypeConverter &converter,
                                   MLIRContext *context)
      : ConvertAnyContextAware(converter, context, /*benefit=*/0) {
    setDebugName("ConvertAnyAddingMaterializedAttr");
  }

  LogicalResult finalizeOpModification(
      Operation *op,
      ContextAwareConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] { setMaterializedAttr(op); });
    return success();
  };
};

class ConvertAssignLayout
    : public ContextAwareOpConversionPattern<tensor_ext::AssignLayoutOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor_ext::AssignLayoutOp>::ContextAwareOpConversionPattern;

  Value expandDims(Value value, LayoutAttr layout,
                   ImplicitLocOpBuilder &b) const {
    RankedTensorType dataSemanticType = cast<RankedTensorType>(value.getType());
    tensor_ext::AlignmentAttr alignment = layout.getAlignment();

    // It's a bit weird, but to make an expand shape op we have to group the
    // output indices in dataSemanticType.getRank() many groups where the 1's
    // are all grouped with axes from the dataSemanticType. But the 1's can
    // show up before or after the data semantic tensor's dims, so we
    // eagerly consume unit dims before and after each data semantic dim.
    SmallVector<int64_t> newSizes;
    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices nextGroup;
    int64_t ciphertextIndex = 0, groupIndex = 0;
    while (groupIndex < dataSemanticType.getRank()) {
      // Process all the unit dims.
      while (containsDim(alignment.getInsertedDims(), ciphertextIndex)) {
        newSizes.push_back(1);
        nextGroup.push_back(ciphertextIndex);
        ++ciphertextIndex;
      }

      // Now process exactly one data dim.
      newSizes.push_back(dataSemanticType.getDimSize(groupIndex));
      nextGroup.push_back(ciphertextIndex);
      ++ciphertextIndex;

      // Now process any more unit dims.
      while (containsDim(alignment.getInsertedDims(), ciphertextIndex)) {
        newSizes.push_back(1);
        nextGroup.push_back(ciphertextIndex);
        ++ciphertextIndex;
      }

      reassociation.push_back(nextGroup);
      nextGroup.clear();
      ++groupIndex;
    }
    RankedTensorType expandedType =
        RankedTensorType::get(newSizes, dataSemanticType.getElementType());
    auto expandOp =
        b.create<tensor::ExpandShapeOp>(expandedType, value, reassociation);
    setMaterializedAttr(expandOp);
    expandOp->setAttr(kLayoutAttrName, layout);
    return expandOp.getResult();
  }

  Value applyPadding(Value value, LayoutAttr layout,
                     ImplicitLocOpBuilder &b) const {
    RankedTensorType dataSemanticType = cast<RankedTensorType>(value.getType());
    tensor_ext::AlignmentAttr alignment = layout.getAlignment();
    // Note padding is asserted to be present, and paddingValue is enforced
    // to be present whenever padding is present due to attribute verifier.
    auto padValueOp = b.create<arith::ConstantOp>(alignment.getPaddingValue());

    SmallVector<int64_t> newSizes;
    SmallVector<OpFoldResult> lows;
    SmallVector<OpFoldResult> highs;
    for (int i = 0; i < dataSemanticType.getRank(); ++i) {
      newSizes.push_back(dataSemanticType.getDimSize(i) +
                         alignment.getPadding()[i]);
      lows.push_back(b.getIndexAttr(0));
      highs.push_back(b.getIndexAttr(alignment.getPadding()[i]));
    }
    RankedTensorType expandedType =
        RankedTensorType::get(newSizes, dataSemanticType.getElementType());
    auto padOp = b.create<tensor::PadOp>(expandedType, value, lows, highs,
                                         padValueOp, /*nofold=*/false);

    setMaterializedAttr(padOp);
    setMaterializedAttr(padValueOp);
    padOp->setAttr(kLayoutAttrName, layout);
    b.setInsertionPointAfter(padOp);
    return padOp.getResult();
  }

  FailureOr<Value> maybeReplicateAlongAxis(tensor_ext::AssignLayoutOp op,
                                           Value value, int axis,
                                           int64_t outputAxisSize,
                                           ImplicitLocOpBuilder &b) const {
    RankedTensorType mostRecentType = cast<RankedTensorType>(value.getType());
    int64_t dataDimSize = mostRecentType.getDimSize(axis);

    if (outputAxisSize % dataDimSize != 0 &&
        dataDimSize % outputAxisSize != 0) {
      auto diag = op.emitError()
                  << "Before replication, tensor size must divide or be a "
                     "multiple of data "
                     "size, or else repetition will not make sense!";
      diag.attachNote()
          << "For dim " << axis << ", target dim size was " << outputAxisSize
          << ", but input size (after optional dim insertion and padding) was "
          << dataDimSize;
      return diag;
    }

    if (dataDimSize < outputAxisSize) {
      // Concatenate appropriately
      SmallVector<int64_t> newSizes =
          SmallVector<int64_t>(mostRecentType.getShape());
      newSizes[axis] = outputAxisSize;
      RankedTensorType expandedShape =
          RankedTensorType::get(newSizes, mostRecentType.getElementType());

      int64_t numIters = outputAxisSize / dataDimSize;
      SmallVector<Value> repeatedInputs(numIters, value);
      auto concatOp = b.create<tensor::ConcatOp>(op.getLoc(), expandedShape,
                                                 /*axis=*/axis, repeatedInputs);
      setMaterializedAttr(concatOp);
      concatOp->setAttr(kLayoutAttrName, op.getLayout());
      return concatOp.getResult();
    }
    return value;
  }

  LogicalResult matchAndRewrite(
      tensor_ext::AssignLayoutOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // This pattern is different because its inputs do not have a layout,
    // so the type converter fails to find attributes for the input types.
    // Instead, the result's layout is defined by this op.
    //
    // The input hence needs to be reshaped to fit the output layout, and
    // by default we assume all values are repeated cyclically according
    // to the layout.
    RankedTensorType dataSemanticType = op.getResult().getType();
    RankedTensorType ciphertextSemanticType = cast<RankedTensorType>(
        getTypeConverter()->convertType(dataSemanticType, op.getLayout()));
    LLVM_DEBUG(llvm::dbgs() << "Converting AssignLayoutOp to use result type "
                            << ciphertextSemanticType << "\n");
    Value input = adaptor.getTensor();
    LayoutAttr layout = op.getLayout();

    // Not all aspects of a replication attribute may be applied. In some rare
    // cases, the input type may already be materialized and no work is
    // required. So this tracks the value that is the result of the most
    // recently applied operation in the process, and the final output value to
    // replace this op with.
    Value mostRecentOutput = input;

    // Apply the semantics of the replication attribute in order before
    // applying the layout.
    tensor_ext::AlignmentAttr alignment = layout.getAlignment();

    if (alignment) {
      // 1. Insert unit dimensions via tensor.expand_shape
      if (alignment.getInsertedDims() && !alignment.getInsertedDims().empty()) {
        mostRecentOutput = expandDims(mostRecentOutput, layout, b);
      }

      // 2. Add padding to the end of each axis via tensor.pad
      if (alignment.getPadding() && !alignment.getPadding().empty()) {
        mostRecentOutput = applyPadding(mostRecentOutput, layout, b);
      }

      // 3. Replicate the input tensor along each axis via tensor.concat
      for (int i = 0; i < alignment.getOut().size(); ++i) {
        FailureOr<Value> res = maybeReplicateAlongAxis(
            op, mostRecentOutput, i, alignment.getOut()[i], b);
        if (failed(res)) return res;
        mostRecentOutput = res.value();
      }
    }

    RankedTensorType mostRecentType =
        cast<RankedTensorType>(mostRecentOutput.getType());

    // At this point, we could try to guarantee that the replicated data tensor
    // has the same number of elements as the ciphertext tensor, but in general
    // this is not required. You could just waste slots, though there is a
    // concern that some kernels that rely on replication may not work as
    // expected. So in this case we emit a warning.
    LLVM_DEBUG({
      if (mostRecentType.getNumElements() !=
          ciphertextSemanticType.getNumElements()) {
        op.emitWarning()
            << "Data type (after replication and padding) " << mostRecentType
            << " has fewer entries than ciphertext type "
            << ciphertextSemanticType
            << ". This may indicate unused slots, or may lead to unexpected "
               "behavior for some kernels that require data replication to "
               "operate properly.";
      }
    });

    // 4. Apply the layout
    if (!layout.getMap().isIdentity()) {
      // Materialize encoding via linalg.generic.
      //
      // Nb., rather than use tensor.empty(), start with constant zeros which
      // plays better with secret.generic lowerings. This implies that any
      // unused values in the layout will default to zero, which seems both
      // like a safe default and the kind of thing that a user could
      // unexpectedly become dependent on.
      auto emptyOp = b.create<mlir::arith::ConstantOp>(
          b.getZeroAttr(ciphertextSemanticType));
      setMaterializedAttr(emptyOp);
      emptyOp->setAttr(kLayoutAttrName, layout);

      SmallVector<utils::IteratorType> iteratorTypes(
          op.getLayout().getMap().getNumDims(), utils::IteratorType::parallel);
      SmallVector<AffineMap> indexingMaps = {
          // The first map corresponds to how the iteration indices map to the
          // input tensor indices. This is the identity because the loop is
          // mapping the input values to ciphertext slots.
          AffineMap::getMultiDimIdentityMap(layout.getMap().getNumDims(),
                                            op.getContext()),
          // The first map is the actual layout, mapping input tensor indices
          // to ciphertext slots.
          layout.getMap()};
      auto materializeLayoutOp = b.create<linalg::GenericOp>(
          /*resultTypes=*/emptyOp.getResult().getType(),
          /*inputs=*/mostRecentOutput,
          /*outputs=*/emptyOp.getResult(), indexingMaps, iteratorTypes,
          /*bodyBuilder=*/
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            // Do nothing, which just assigns the input to the output slot.
            auto yieldOp =
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
            setMaterializedAttr(yieldOp);
          });

      setMaterializedAttr(materializeLayoutOp);
      materializeLayoutOp->setAttr(kLayoutAttrName,
                                   op->getAttr(kLayoutAttrName));
      mostRecentOutput = materializeLayoutOp.getResult(0);
    }

    if (mostRecentOutput == input) {
      setAttributeAssociatedWith(mostRecentOutput, kLayoutAttrName, layout);
      LLVM_DEBUG(llvm::dbgs()
                 << "No materialization needed, passing input through\n");
    }

    rewriter.replaceOp(op, mostRecentOutput);
    return success();
  };
};

// Return the first output index not mapped to by the partial permutation.
int64_t getMinUnusedTarget(llvm::ArrayRef<int64_t> perm) {
  std::vector<int64_t> unmappedOutputsVector(perm.size());
  std::iota(unmappedOutputsVector.begin(), unmappedOutputsVector.end(), 0);
  std::set<int64_t> unmappedOutputs(unmappedOutputsVector.begin(),
                                    unmappedOutputsVector.end());
  for (int64_t target : perm) {
    if (target != kUnset) {
      unmappedOutputs.erase(target);
    }
  }

  if (unmappedOutputs.empty()) {
    return -1;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Unmapped outputs: ";
    for (int64_t i : unmappedOutputs) {
      llvm::dbgs() << i << " ";
    }
    llvm::dbgs() << "\n";
  });

  return *unmappedOutputs.begin();
}

// Return the first unused input index not mapped from by the partial
// permutation.
int64_t getMinUnusedInput(llvm::ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < perm.size(); ++i) {
    if (perm[i] == kUnset) return i;
  }
  return -1;
}

class ConvertConvertLayout
    : public ContextAwareOpConversionPattern<tensor_ext::ConvertLayoutOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor_ext::ConvertLayoutOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor_ext::ConvertLayoutOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    RankedTensorType dataSemanticType = op.getTensor().getType();
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getTensor().getType());
    LLVM_DEBUG(llvm::dbgs()
               << "ConvertConvertLayout: dataSemanticType=" << dataSemanticType
               << ", ciphertextSemanticType=" << ciphertextSemanticType
               << "\n");
    LayoutAttr fromLayout = op.getFromLayout();
    LayoutAttr toLayout = op.getToLayout();

    // TODO(#1542): support multi-packed ciphertexts
    if (ciphertextSemanticType.getRank() != 1) {
      return op.emitError()
             << "Does not support packing into multiple ciphertexts yet";
    }

    int64_t numSlots = ciphertextSemanticType.getShape().back();
    SmallVector<int64_t> permutation(numSlots, kUnset);

    // The algorithm here allows the permutation to be built up "cyclically"
    // in the following sense: after the original layout permutation is
    // exhausted, we then repeat that layout permutation with offsets
    // corresponding to the first unused input and target indices.
    //
    // Example: tensor<4xi16> in a ciphertext of size 16,
    //
    //  Converting layout d0 -> d0 to d0 -> 4*d0:
    //
    //   (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3)
    //
    //     maps to the corresponding layout
    //
    //   (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
    //
    int64_t minUnusedTarget = 0;
    int64_t minUnusedInput = 0;
    while (minUnusedInput != -1) {
      LLVM_DEBUG({
        if (minUnusedInput > 0) {
          llvm::dbgs() << "Repeating for cyclic repetition with"
                       << " minUnusedInput=" << minUnusedInput
                       << " minUnusedTarget=" << minUnusedTarget << "\n";
        }
      });
      IndexTupleConsumer evaluateNextIndex =
          [&](const std::vector<int64_t> &indices) {
            SmallVector<int64_t> fromResults;
            SmallVector<int64_t> toResults;
            evaluateStatic(fromLayout.getMap(), indices, fromResults);
            evaluateStatic(toLayout.getMap(), indices, toResults);
            int64_t input =
                (minUnusedInput + fromResults[fromResults.size() - 1]) %
                numSlots;
            int64_t output =
                (minUnusedTarget + toResults[toResults.size() - 1]) % numSlots;
            permutation[input] = output;
          };
      iterateIndices(dataSemanticType.getShape(), evaluateNextIndex);
      minUnusedTarget = getMinUnusedTarget(permutation);
      minUnusedInput = getMinUnusedInput(permutation);
    }

    auto permuteOp = rewriter.create<tensor_ext::PermuteOp>(
        op.getLoc(), adaptor.getTensor(),
        rewriter.getI64TensorAttr(permutation));
    permuteOp->setAttr(kLayoutAttrName, op->getAttr(kLayoutAttrName));
    setMaterializedAttr(permuteOp);
    rewriter.replaceOp(op, permuteOp);
    return success();
  };
};

// If the mapping is a partial rotation, return the rotation shift amount.
std::optional<int64_t> tryDetectPartialRotation(
    ::llvm::ArrayRef<int64_t> perm) {
  std::optional<int64_t> rotation = std::nullopt;
  for (int64_t i = 0; i < perm.size(); ++i) {
    int64_t input = i;
    int64_t output = perm[i];
    if (output == kUnset) continue;
    // We rotate left in this codebase, so invert normal output - input
    int64_t shiftAmount = -(output - input);
    if (!rotation.has_value()) {
      rotation = shiftAmount;
    } else if (shiftAmount != rotation.value()) {
      return std::nullopt;
    }
  }
  return rotation;
}

// Extend a partial permutation to a full permutation in an FHE-friendly way.
void extendPermutationGreedily(::llvm::MutableArrayRef<int64_t> perm) {
  std::set<int64_t> unmappedInputs;

  // Start with values 0..n-1 and remove when found in the permutation
  std::vector<int64_t> unmappedOutputsVector(perm.size());
  std::iota(unmappedOutputsVector.begin(), unmappedOutputsVector.end(), 0);
  std::set<int64_t> unmappedOutputs(unmappedOutputsVector.begin(),
                                    unmappedOutputsVector.end());

  for (int64_t i = 0; i < perm.size(); ++i) {
    if (perm[i] == kUnset) {
      unmappedInputs.insert(i);
    } else {
      unmappedOutputs.erase(perm[i]);
    }
  }

  // Set iteration is in sorted order, so we're mapping each unused input to
  // the first output index that hasn't been mapped to yet.
  for (const auto &[input, output] :
       llvm::zip(unmappedInputs, unmappedOutputs)) {
    perm[input] = output;
  }
}

// Extend a partial permutation to a full permutation in an FHE-friendly way.
//
// FHE-friendly means that the output permutation should lower to a small shift
// network. For example, if the permutation can be extended to a single
// rotation, it should be.
//
// The input partialPermutation must already be correctly sized (size n for a
// permutation on 1..n). Unset entries of the permutation are indicated by
// kUnset.
void extendPartialPermutation(MutableArrayRef<int64_t> partialPermutation) {
  // If the partially set entries correspond to a single rotation, extend it.
  std::optional<int64_t> rotation =
      tryDetectPartialRotation(partialPermutation);
  if (rotation.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "Detected partial rotation of offset "
                            << rotation.value() << "\n");
    for (int64_t i = 0; i < partialPermutation.size(); ++i) {
      if (partialPermutation[i] == kUnset) {
        int64_t target = i - rotation.value();
        if (target < 0) target += partialPermutation.size();
        partialPermutation[i] = target;
      }
    }
    return;
  }

  // Otherwise, try to fill in the unset entries greedily.
  extendPermutationGreedily(partialPermutation);
}

class ConvertLinalgReduce
    : public ContextAwareOpConversionPattern<linalg::ReduceOp> {
 public:
  using ContextAwareOpConversionPattern<
      linalg::ReduceOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::ReduceOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    // Ensure the reduction op is single addition or multiplication, otherwise
    // there is no kernel.
    Block *body = op.getBlock();
    if (body->getOperations().size() != 2) {
      return op.emitError(
          "linalg.reduce only supported with a single reduction operation");
    }

    // TODO(#1543): support multi-dimension reductions
    if (op.getDimensions().size() != 1) {
      return op.emitError(
          "linalg.reduce only supported with a single reduction dimension");
    }

    if (!op.isSingleInputOutput()) {
      return op.emitError(
          "linalg.reduce only supported with a single reduction dimension");
    }

    Operation *innerOp = &body->getOperations().front();
    if (!isa<arith::AddFOp, arith::MulFOp, arith::AddIOp, arith::MulIOp>(
            innerOp)) {
      return op.emitError()
             << "linalg.reduce only supported with a single addition or "
                "multiplication operation, but found: "
             << innerOp->getName();
    }

    // Example: To reduce a tensor<32x32xi16> along dimension 0 using addition,
    // that is laid out in a ciphertext-semantic tensor<1024xi16> via some
    // mapping, we need to compute the permutations required to align the
    // inputs along rows.
    //
    // If the layouts are chosen well, for this example if the layout is
    // row-major, then that permutation corresponds to a simple set of
    // rotations.
    //
    // TODO(#1542): support multi-packed ciphertexts

    RankedTensorType dataSemanticType =
        cast<RankedTensorType>(op.getInputs()[0].getType());
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getInputs()[0].getType());
    FailureOr<Attribute> layoutFetchResult =
        getTypeConverter()->getContextualAttr(adaptor.getInputs()[0]);
    if (failed(layoutFetchResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    LayoutAttr layout = cast<LayoutAttr>(layoutFetchResult.value());

    // See DestinationStyleOpInterface: 1-1 relationship between inits and op
    // results, but the init is type converted already and is the starting
    // point for the reduction kernel implementation.
    Value init = adaptor.getInits()[0];
    Value input = adaptor.getInputs()[0];

    LLVM_DEBUG(llvm::dbgs()
               << "ConvertLinalgReduce:\n"
               << "\n  - data semantic type: " << dataSemanticType
               << "\n  - ciphertext semantic type: " << ciphertextSemanticType
               << "\n  - layout: " << layout << "\n  - dimensions: "
               << llvm::join(llvm::map_range(
                                 op.getDimensions(),
                                 [](int64_t d) { return std::to_string(d); }),
                             ", ")
               << "\n  - op: " << innerOp->getName() << "\n  - init: " << init
               << "\n\n");

    int64_t reductionDim = op.getDimensions()[0];
    int64_t reductionDimSize = dataSemanticType.getShape()[reductionDim];

    // In the example from above (row-major <32x32> -> <1024>), all values
    // in the reduced dimension need to map to the same slot. E.g.,
    //
    //   (0, 0), (1, 0), (2, 0), ... (31, 0)
    //
    // should map to slot (0) in each term of the sum, while
    //
    //   (0, 1), (1, 1), (2, 1), ... (31, 1)
    //
    // should all map to slot (1). Viewing this with a different iteration
    // order, there is one summand for each entry of the reduced dimension, and
    // in that summand we must map
    //
    //   (i, 0), (i, 1), (i, 2), ... (i, 31)
    //
    // to
    //
    //   (0, 1, 2, ..., 31)
    //
    // For this example row-major layout, the entry (i, 3) maps to 32*i + 3,
    // and we need it to map to 3. So this adds a requirement that the
    // permutation maps 32*i + 3 -> 3. This can be calculated generically by
    // evaluating the layout at (i, 3) and mapping it to the static index tuple
    // given by the non-iterating dimensions of the tensor (dim 1 in the
    // example, which has a value of 3) in this step of the iteration.
    //
    // The mechanism above populates one part of a larger permutation of the
    // ciphertext-semantic tensor, and then we have to find a way to extend
    // that to a permutation of the whole tensor in a way that will produce
    // simple shift networks. For now we identify the case that the partial
    // permutation can be extended to a simple shift, and then we fill in the
    // rest of the permutation according to that shift. Otherwise, we fill in
    // the rest of the permutation in a greedy order.
    //
    // TODO(#521): Extend rotate-and-reduce so it can be run after this kernel
    // and find additional rotation optimizations.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value result = init;
    for (int64_t dimIndex = 0; dimIndex < reductionDimSize; ++dimIndex) {
      SmallVector<int64_t> permutation(ciphertextSemanticType.getNumElements(),
                                       kUnset);
      SmallVector<int64_t> fixedIndices = {reductionDim};
      SmallVector<int64_t> fixedValues = {dimIndex};

      // For each entry with the reduced index fixed, populate the permutation
      // with the desired partial mapping.
      IndexTupleConsumer evaluateNextIndex =
          [&](const std::vector<int64_t> &indices) {
            SmallVector<int64_t> results;
            evaluateStatic(layout.getMap(), indices, results);

            // Since we are reducing along a dimension, the input that gives us
            // the desired output slot should be the slot that the layout maps
            // the first entry of the reduced dimension to.
            //
            // From the running example tensor<32x32> -> tensor<1024> row-major
            // layout reducing dim 0, if we're at entry dimIndex = 3, then we're
            // saying that all entries (3, j) should map to (0, j) so that all
            // the values (0, j), (1, j), (2, j), ... (31, j) are aligned.
            std::vector<int64_t> inputsForDesiredResults(indices);
            for (int64_t fixedIndex : fixedIndices) {
              inputsForDesiredResults[fixedIndex] = 0;
            }
            SmallVector<int64_t> desiredResults;
            evaluateStatic(layout.getMap(), inputsForDesiredResults,
                           desiredResults);

            // The last dimension of the layout output is the ciphertext
            // dimension, and it contains the slot that the entry is mapped to.
            permutation[results[results.size() - 1]] =
                desiredResults[desiredResults.size() - 1];
          };

      iterateIndices(dataSemanticType.getShape(), evaluateNextIndex,
                     fixedIndices, fixedValues);
      extendPartialPermutation(permutation);

      auto permuteOp = b.create<tensor_ext::PermuteOp>(
          input, b.getI64TensorAttr(permutation));

      SmallVector<Value> operands = {result, permuteOp};
      SmallVector<Type> newResultTypes = {permuteOp.getType()};
      Operation *nextOp = rewriter.create(
          OperationState(op->getLoc(), innerOp->getName().getStringRef(),
                         operands, newResultTypes));
      permuteOp->setAttr(kLayoutAttrName, op->getAttr(kLayoutAttrName));
      nextOp->setAttr(kLayoutAttrName, op->getAttr(kLayoutAttrName));
      setMaterializedAttr(permuteOp);
      setMaterializedAttr(nextOp);
      result = nextOp->getResult(0);
    }

    // TODO(#1591): post-process the layout properly for padding
    rewriter.replaceOp(op, result);
    return success();
  }
};

bool isLayoutSquatDiagonal(RankedTensorType inputType,
                           RankedTensorType outputType,
                           const AffineMap &layout) {
  // Squat diagonal forces (i, j) -> (j % n, (i+j) % m) where (n, m) are the
  // dimensions of the output matrix.
  if (outputType.getRank() != 2 || inputType.getRank() != 2) return false;

  int64_t n = outputType.getDimSize(0);
  int64_t m = outputType.getDimSize(1);
  AffineExpr i, j;
  bindDims(inputType.getContext(), i, j);
  AffineMap expected =
      AffineMap::get(2, 0, {j % n, (i + j) % m}, inputType.getContext());

  auto simplified = simplifyAffineMap(layout);
  LLVM_DEBUG(llvm::dbgs() << "isLayoutSquatDiagonal: " << "simplified="
                          << simplified << " expected=" << expected << "\n");
  return simplified == expected;
}

struct ConvertLinalgMatvec
    : public ContextAwareOpConversionPattern<linalg::MatvecOp> {
 public:
  using ContextAwareOpConversionPattern<
      linalg::MatvecOp>::ContextAwareOpConversionPattern;

  LayoutAttr getLayoutAttr(Value value) const {
    auto layoutLookup = getTypeConverter()->getContextualAttr(value);
    if (failed(layoutLookup)) {
      return nullptr;
    }
    return cast<LayoutAttr>(layoutLookup.value());
  }

  bool supportsHaleviShoup(linalg::MatvecOp op, OpAdaptor adaptor) const {
    Value matrix = op.getInputs()[0];
    Value vector = op.getInputs()[1];
    auto matrixType = cast<RankedTensorType>(matrix.getType());
    auto vectorType = cast<RankedTensorType>(vector.getType());
    auto materializedMatrixType =
        cast<RankedTensorType>(adaptor.getInputs()[0].getType());
    auto materializedVectorType =
        cast<RankedTensorType>(adaptor.getInputs()[1].getType());

    // If one of these dimensions is not a power of two, then we can't do
    // the Halevi-Shoup or Squat Packing Matrix Multiplication conversion.
    auto dimensions = matrixType.getShape();
    int64_t numRows = dimensions[0];
    int64_t numCols = dimensions[1];
    if (!isPowerOfTwo(numRows) || !isPowerOfTwo(numCols)) {
      return false;
    }

    LayoutAttr matrixLayout = getLayoutAttr(matrix);
    LayoutAttr vectorLayout = getLayoutAttr(vector);
    bool isSquatDiagonal = isLayoutSquatDiagonal(
        matrixType, materializedMatrixType, matrixLayout.getMap());
    bool isRowMajor = isLayoutRowMajor(vectorType, materializedVectorType,
                                       vectorLayout.getMap());

    LLVM_DEBUG(llvm::dbgs()
               << "supportsHaleviShoup: " << "isSquatDiagonal="
               << isSquatDiagonal << " isRowMajor=" << isRowMajor << "\n");

    // TODO(#1578): If the matrix has more rows than columns, what kernel
    // should be used?
    bool dimensionsCompatible = numRows <= numCols;
    return isSquatDiagonal && isRowMajor && dimensionsCompatible;
  }

  void haleviShoupKernel(
      linalg::MatvecOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value result = adaptor.getOutputs()[0];
    Value packedMatrix = adaptor.getInputs()[0];
    Value packedVector = adaptor.getInputs()[1];
    auto packedMatrixType = cast<RankedTensorType>(packedMatrix.getType());
    auto packedVectorType = cast<RankedTensorType>(packedVector.getType());
    Type elementType = packedVectorType.getElementType();
    int64_t numRotations = packedMatrixType.getShape()[0];
    auto layoutAttr = cast<LayoutAttr>(op->getAttr(kLayoutAttrName));

    StringRef mulOpName =
        isa<IntegerType>(elementType) ? "arith.muli" : "arith.mulf";
    StringRef addOpName =
        isa<IntegerType>(elementType) ? "arith.addi" : "arith.addf";
    // In each loop iteration we rotate by 1 more, which minimizes the number
    // of rotation keys needed, but also adds a serial dependency to the order
    // of operations in the loop.
    //
    // TODO(#744): determine how to balance the tradeoff of having many
    // rotation keys vs overall latency for doing some rotations in parallel
    // (plus hoisting).
    //
    // TODO(#1569): consider emitting linalg ops and/or loops for the rotations
    // and reductions. Maybe branch on a pass option and support all three?
    // Need to determine how noise analysis will handle an ambiguous linalg op
    // before doing this by default.
    Value accumulator = result;
    Value incrementallyRotatedVector = packedVector;
    auto constantOne = b.create<arith::ConstantIntOp>(1, 64);
    for (int index = 0; index < numRotations; ++index) {
      // construct vector.rotate(i)
      auto rotateOp = b.create<tensor_ext::RotateOp>(incrementallyRotatedVector,
                                                     constantOne);

      // get the corresponding element of the ciphertext tensor,
      // which is row i
      SmallVector<OpFoldResult> offsets = {b.getIndexAttr(index),
                                           b.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {
          b.getIndexAttr(1), b.getIndexAttr(packedMatrixType.getShape()[1])};
      SmallVector<OpFoldResult> strides = {b.getIndexAttr(1),
                                           b.getIndexAttr(1)};
      auto extractRowOp = b.create<tensor::ExtractSliceOp>(
          packedVectorType, packedMatrix, offsets, sizes, strides);

      Operation *mulOp = b.create(
          OperationState(op->getLoc(), mulOpName,
                         {rotateOp.getResult(), extractRowOp.getResult()},
                         {packedVectorType}));
      Operation *addOp = b.create(OperationState(
          op->getLoc(), addOpName, {accumulator, mulOp->getResult(0)},
          {packedVectorType}));

      setMaterializedAttr(rotateOp);
      setMaterializedAttr(extractRowOp);
      setMaterializedAttr(mulOp);
      setMaterializedAttr(addOp);

      accumulator = addOp->getResult(0);
      incrementallyRotatedVector = rotateOp.getResult();
    }
    // Only the last op will need its layout set for later ops to reference.
    setAttributeAssociatedWith(accumulator, kLayoutAttrName, layoutAttr);

    int64_t matrixNumRows = packedMatrixType.getShape()[0];
    int64_t matrixNumCols = packedMatrixType.getShape()[1];

    Value summedShifts = accumulator;
    if (matrixNumRows == matrixNumCols) {
      rewriter.replaceOp(op, summedShifts);
      return;
    }

    // else, necessarily matrixNumRows < matrixNumCols due to the precondition
    // applied earlier. This is the post-processing partial-rotate-and-reduce
    // step required for squat-diagonal packing.

    int64_t numShifts = (int64_t)(log2(matrixNumCols) - log2(matrixNumRows));
    int64_t shift = matrixNumCols / 2;

    for (int64_t i = 0; i < numShifts; ++i) {
      auto rotateOp = b.create<tensor_ext::RotateOp>(
          summedShifts, b.create<arith::ConstantIntOp>(shift, 64));
      setMaterializedAttr(rotateOp);
      auto *addOp = b.create(OperationState(
          op->getLoc(), addOpName, {summedShifts, rotateOp.getResult()},
          {rotateOp.getResult().getType()}));
      summedShifts = addOp->getResult(0);
      shift /= 2;
    }

    // The result now has the values in the first n entries of the packed
    // vector, and the remaining entries are naturally replicated copies
    // of the first n entries. So if the output layout does not require
    // a special padding, we can stop here.
    if (!layoutAttr.getAlignment() ||
        layoutAttr.getAlignment().getPadding().empty()) {
      // TODO(#1569): also hit this branch if the padding value is dont_care
      rewriter.replaceOp(op, summedShifts);
      return;
    }

    // Otherwise, the output layout requires a particular padding, and we
    // need to force the replicated values to be zero. This is done by
    // applying a plaintext-ciphertext mask.
    TypedAttr padAttr =
        DenseElementsAttr::get(cast<ShapedType>(result.getType()),
                               layoutAttr.getAlignment().getPaddingValue());
    auto zeroOp = b.create<arith::ConstantOp>(result.getType(), padAttr);

    // insert a slice of 1's in the first n of the zeros tensor to make a mask
    SmallVector<int64_t> prefixShape = {matrixNumRows};
    RankedTensorType prefixType =
        RankedTensorType::get(prefixShape, elementType);
    auto oneOp =
        b.create<arith::ConstantOp>(prefixType, b.getOneAttr(prefixType));
    auto createMaskOp = b.create<tensor::InsertSliceOp>(
        oneOp, zeroOp, ArrayRef<Value>{}, ArrayRef<Value>{}, ArrayRef<Value>{},
        /*offsets=*/ArrayRef<int64_t>{0},
        /*sizes=*/ArrayRef{matrixNumRows}, /*strides=*/ArrayRef<int64_t>{1});
    auto *applyMaskOp = b.create(OperationState(op->getLoc(), mulOpName,
                                                {summedShifts, createMaskOp},
                                                {summedShifts.getType()}));

    setMaterializedAttr(zeroOp);
    setMaterializedAttr(oneOp);
    setMaterializedAttr(createMaskOp);
    setMaterializedAttr(applyMaskOp);
    applyMaskOp->setAttr(kLayoutAttrName, layoutAttr);

    rewriter.replaceOp(op, applyMaskOp);
  }

  LogicalResult matchAndRewrite(
      linalg::MatvecOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    Value matrix = op.getInputs()[0];
    Value vector = op.getInputs()[1];

    if (!getLayoutAttr(matrix) || !getLayoutAttr(vector)) {
      return op.emitError() << "missing layout attribute for matrix or vector";
    }

    if (supportsHaleviShoup(op, adaptor)) {
      haleviShoupKernel(op, adaptor, rewriter);
      return success();
    }

    // TODO(#1589): implement row-major naive matvec kernel
    return op.emitError() << "unsupported layout for matrix in matvec: "
                          << getLayoutAttr(matrix);
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
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isa<ModuleOp>(op) || hasMaterializedAttr(op);
    });

    patterns.add<ConvertFunc, ConvertGeneric,
                 // tensor_ext ops
                 ConvertAssignLayout, ConvertConvertLayout,
                 // linalg ops
                 ConvertLinalgReduce, ConvertLinalgMatvec,
                 // default
                 ConvertAnyAddingMaterializedAttr>(typeConverter, context);

    if (failed(applyContextAwarePartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    // Decompose tensor.concat into repeated tensor.insert_slice ops.
    // Note ConvertAssignLayout generates tensor.concat
    RewritePatternSet cleanupPatterns2(context);
    tensor::populateDecomposeTensorConcatPatterns(cleanupPatterns2);
    walkAndApplyPatterns(module, std::move(cleanupPatterns2));

    clearAttrs(module, kLayoutAttrName);
    clearAttrs(module, kMaterializedAttrName);
  }
};

}  // namespace heir
}  // namespace mlir
