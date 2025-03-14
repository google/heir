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

using tensor_ext::LayoutAttr;

auto &kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
auto &kMaterializedAttrName = "tensor_ext.layout_materialized";
auto &kOriginalTypeAttrName =
    tensor_ext::TensorExtDialect::kOriginalTypeAttrName;

// An unset value of a permutation as it's being built up.
static constexpr int kUnset = -1;

#define GEN_PASS_DEF_CONVERTTOCIPHERTEXTSEMANTICS
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h.inc"

bool isPowerOfTwo(int64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

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
    addConversion([&](secret::SecretType type,
                      AffineMapAttr attr) -> std::optional<Type> {
      auto innerType = type.getValueType();
      auto rankedTensorType = dyn_cast<RankedTensorType>(innerType);
      if (!rankedTensorType) return std::nullopt;

      auto convertedInnerType = materializeLayout(rankedTensorType, attr);
      if (failed(convertedInnerType)) return std::nullopt;

      return secret::SecretType::get(convertedInnerType.value());
    });
    addConversion(
        [&](RankedTensorType type, AffineMapAttr attr) -> std::optional<Type> {
          return materializeLayout(type, attr);
        });
  }

  FailureOr<Type> materializeLayout(RankedTensorType type,
                                    AffineMapAttr attr) const {
    AffineMap layout = attr.getValue();
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
  using ContextAwareFuncConversion::ContextAwareFuncConversion;

  ConvertFunc(const ContextAwareTypeConverter &converter, MLIRContext *context)
      : ContextAwareFuncConversion(converter, context) {}

  LogicalResult finalizeFuncOpModification(
      func::FuncOp op, ArrayRef<Type> oldArgTypes,
      ArrayRef<Type> oldResultTypes, PatternRewriter &rewriter) const override {
    // Replace layout arg attrs with secret.original_type arg attrs This is
    // necessary so that later encoding/decoding functions can know what the
    // original type of the tensor was and how it was encoded.
    //
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
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(getTypeConverter()->convertType(
            op.getResult().getType(), op.getLayout()));
    LLVM_DEBUG(llvm::dbgs() << "Converting AssignLayoutOp to use result type "
                            << ciphertextSemanticType << "\n");
    RankedTensorType dataSemanticType = op.getTensor().getType();
    Value input = adaptor.getTensor();
    AffineMap layout = op.getLayout().getMap();

    // FIXME: what will happen if the output size is bigger than the input size?
    if (ciphertextSemanticType.getRank() != 1) {
      // Materialize encoding via linalg.generic (is there a named op that can
      // permute indices like this?)
      //
      // Nb., rather than use tensor.empty(), start with constant zeros which
      // plays better with secret.generic lowerings.
      auto emptyOp = b.create<mlir::arith::ConstantOp>(
          b.getZeroAttr(ciphertextSemanticType));
      setMaterializedAttr(emptyOp);
      emptyOp->setAttr(kLayoutAttrName, op.getLayout());

      SmallVector<utils::IteratorType> iteratorTypes(
          op.getLayout().getMap().getNumDims(), utils::IteratorType::parallel);
      SmallVector<AffineMap> indexingMaps = {
          // The first map corresponds to how the iteration indices map to the
          // input tensor indices. This is the identity because the loop is
          // mapping the input values to ciphertext slots.
          AffineMap::getMultiDimIdentityMap(layout.getNumDims(),
                                            op.getContext()),
          // The first map is the actual layout, mapping input tensor indices
          // to ciphertext slots.
          layout};
      auto materializeLayoutOp = b.create<linalg::GenericOp>(
          /*resultTypes=*/emptyOp.getResult().getType(),
          /*inputs=*/input,
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
      rewriter.replaceOp(op, materializeLayoutOp);
      return success();
    }

    if (ciphertextSemanticType == input.getType() && layout.isIdentity()) {
      LLVM_DEBUG(llvm::dbgs() << "AssignLayoutOp can be replaced with input\n");
      if (input.getDefiningOp()) setMaterializedAttr(input.getDefiningOp());
      setAttributeAssociatedWith(input, kLayoutAttrName, op.getLayout());
      rewriter.replaceOp(op, input);
      return success();
    }

    // Otherwise, tile into the ciphertext
    int dataSize = dataSemanticType.getNumElements();
    int ciphertextSize = ciphertextSemanticType.getShape().back();
    // TODO(#704): align tensor sizes when they don't align with num slots
    if (ciphertextSize % dataSize != 0) {
      return op.emitError() << "Ciphertext size must be a multiple of data "
                               "size, or else repetition assumption fails!";
    }

    int64_t numIters = ciphertextSize / dataSize;
    SmallVector<Value> repeatedInputs(numIters, input);
    auto concatOp = b.create<tensor::ConcatOp>(
        op.getLoc(), ciphertextSemanticType, /*axis=*/0, repeatedInputs);
    setMaterializedAttr(concatOp);
    concatOp->setAttr(kLayoutAttrName, op->getAttr(kLayoutAttrName));

    rewriter.replaceOp(op, concatOp);
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
    AffineMap layout =
        cast<AffineMapAttr>(layoutFetchResult.value()).getValue();

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
            evaluateStatic(layout, indices, results);

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
            evaluateStatic(layout, inputsForDesiredResults, desiredResults);

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

    rewriter.replaceOp(op, result);
    return success();
  };
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

    return isSquatDiagonal && isRowMajor;
  }

  void haleviShoupKernel(
      linalg::MatvecOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value result = op.getOutputs()[0];
    Value packedMatrix = adaptor.getInputs()[0];
    Value packedVector = adaptor.getInputs()[1];
    auto packedMatrixType = cast<RankedTensorType>(packedMatrix.getType());
    auto packedVectorType = cast<RankedTensorType>(packedVector.getType());
    int64_t numRotations = packedVectorType.getShape()[0];
    Attribute layoutAttr = op->getAttr(kLayoutAttrName);

    // Two iter args: one for the accumulated result, and one for the
    // iteratively rotated vector; in each loop we rotate by 1 more, which
    // minimizes the number of rotation keys needed, but also adds a serial
    // dependency to the order of operations in the loop.
    //
    // TODO(#744): should we instead use many rotation keys and let a later
    // pass determine how to optimize this?
    SmallVector<Value> iterArgs({result, packedVector});
    auto forOp =
        b.create<mlir::affine::AffineForOp>(0, numRotations, 1, iterArgs);
    setMaterializedAttr(forOp);
    // The for will have two yielded results, and though we only the first
    // result should be queried by the context-aware type converter for its
    // layout attribute when processing later ops, I'm going to set both for
    // consistency.
    OperandAndResultAttrInterface forOpInterface(forOp);
    forOpInterface.setResultAttr(0, kLayoutAttrName, layoutAttr);
    forOpInterface.setResultAttr(1, kLayoutAttrName, layoutAttr);

    b.setInsertionPointToStart(forOp.getBody());
    auto index = forOp.getInductionVar();
    auto accumulator = forOp.getRegionIterArgs()[0];
    auto incrementallyRotatedVector = forOp.getRegionIterArgs()[1];

    // construct vector.rotate(i)
    auto rotateOp = b.create<tensor_ext::RotateOp>(
        incrementallyRotatedVector, b.create<arith::ConstantIntOp>(1, 64));
    setMaterializedAttr(rotateOp);

    // get the corresponding element of the ciphertext tensor,
    // which is row i
    SmallVector<OpFoldResult> offsets = {index, b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {
        b.getIndexAttr(1), b.getIndexAttr(packedMatrixType.getShape()[1])};
    SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
    auto extractRowOp =
        b.create<tensor::ExtractSliceOp>(packedMatrix, offsets, sizes, strides);
    setMaterializedAttr(extractRowOp);

    // The extractRowOp produces a 1xN tensor, but we need a rank-1 tensor, so
    // drop the unit dim. (Is it possible to use ExtractSliceOp to get a rank-1
    // output?)
    SmallVector<mlir::ReassociationIndices> reassociation;
    reassociation.push_back(mlir::ReassociationIndices({0, 1}));
    auto collapseOp = b.create<tensor::CollapseShapeOp>(
        extractRowOp.getResult(), reassociation);
    setMaterializedAttr(collapseOp);

    StringRef mulOpName = isa<IntegerType>(packedVectorType.getElementType())
                              ? "arith.muli"
                              : "arith.mulf";
    StringRef addOpName = isa<IntegerType>(packedVectorType.getElementType())
                              ? "arith.addi"
                              : "arith.addf";
    Operation *mulOp = b.create(OperationState(
        op->getLoc(), mulOpName, {rotateOp.getResult(), collapseOp.getResult()},
        {packedVectorType}));
    Operation *addOp = b.create(
        OperationState(op->getLoc(), addOpName,
                       {accumulator, mulOp->getResult(0)}, {packedVectorType}));
    auto yieldOp = b.create<affine::AffineYieldOp>(
        ValueRange({addOp->getResult(0), rotateOp.getResult()}));

    setMaterializedAttr(mulOp);
    setMaterializedAttr(addOp);
    setMaterializedAttr(yieldOp);

    rewriter.replaceOp(op, forOp.getResult(0));
    // FIXME: add the reduction step to combine partial sums for squat packing
    //
    // FIXME: mark out the first n entries in the squat packing case
    //
    // FIXME: only do squat packing if n < m, otherwise give up?
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

    // Otherwise options are:
    //
    // - insert layout conversion, but the optimizer was supposed to do this
    // - do a naive matvec kernel, if the input is laid out row-major
    //
    // For now, return failure.
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

    // Hoist tensor.concat ops from constant layouts out of the generic
    // RewritePatternSet cleanupPatterns(context);
    // cleanupPatterns.add<secret::HoistPlaintextOps>(context);
    // walkAndApplyPatterns(module, std::move(cleanupPatterns));

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
