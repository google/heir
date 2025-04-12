#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"

#include <cmath>
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
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
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
    // For some reason, directly capturing ciphertextSize here leads to memory
    // corruption on that int. Instead, pass the value to a member variable and
    // query it at call time. I have no idea why C++ does this. Debugging it
    // felt like having a stroke.
    addConversion([&](Type type, Attribute attr) { return std::nullopt; });
    addConversion([this](secret::SecretType type,
                         LayoutAttr attr) -> std::optional<Type> {
      FailureOr<Type> convertedInnerType;
      auto innerType = type.getValueType();

      if (auto rankedTensorType = dyn_cast<RankedTensorType>(innerType)) {
        convertedInnerType =
            materializeLayout(rankedTensorType, attr, getCiphertextSize());
      } else {
        convertedInnerType =
            materializeScalarLayout(innerType, attr, getCiphertextSize());
      }

      if (failed(convertedInnerType)) return std::nullopt;
      return secret::SecretType::get(convertedInnerType.value());
    });
    addConversion(
        [this](RankedTensorType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeLayout(type, attr, getCiphertextSize());
        });
    addConversion(
        [this](IntegerType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeScalarLayout(type, attr, getCiphertextSize());
        });
    addConversion(
        [this](FloatType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeScalarLayout(type, attr, getCiphertextSize());
        });
  }

  int getCiphertextSize() const { return ciphertextSize; }

 private:
  int ciphertextSize;
};

bool hasMaterializedAttr(Operation *op) {
  return op->hasAttr(kMaterializedAttrName);
}

void setMaterializedAttr(Operation *op) {
  op->setAttr(kMaterializedAttrName, UnitAttr::get(op->getContext()));
}

void setMaterializedAttr(ArrayRef<Operation *> ops) {
  for (auto *op : ops) {
    setMaterializedAttr(op);
  }
}

Type maybeExtractSecretType(Type type) {
  if (auto secretType = dyn_cast<secret::SecretType>(type)) {
    return secretType.getValueType();
  }
  return type;
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
                          getContext(), maybeExtractSecretType(oldArgTypes[i]),
                          layoutAttr));
      }

      for (int i = 0; i < op.getNumResults(); ++i) {
        auto layoutAttr =
            dyn_cast_or_null<LayoutAttr>(op.getResultAttr(i, kLayoutAttrName));
        if (!layoutAttr) continue;

        op.setResultAttr(
            i, kOriginalTypeAttrName,
            tensor_ext::OriginalTypeAttr::get(
                getContext(), maybeExtractSecretType(oldResultTypes[i]),
                layoutAttr));
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
  ConvertAssignLayout(const ContextAwareTypeConverter &typeConverter,
                      mlir::MLIRContext *context, int64_t ciphertextSize)
      : ContextAwareOpConversionPattern<tensor_ext::AssignLayoutOp>(
            typeConverter, context),
        ciphertextSize(ciphertextSize) {}

  LogicalResult matchAndRewrite(
      tensor_ext::AssignLayoutOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto res =
        implementAssignLayout(op, ciphertextSize, b, [&](Operation *createdOp) {
          setMaterializedAttr(createdOp);
          createdOp->setAttr(kLayoutAttrName, op.getLayout());
        });
    if (failed(res)) return failure();

    if (res.value() == op.getValue()) {
      setAttributeAssociatedWith(res.value(), kLayoutAttrName, op.getLayout());
      LLVM_DEBUG(llvm::dbgs()
                 << "No materialization needed, passing input through\n");
    }

    rewriter.replaceOp(op, res.value());
    return success();
  };

 private:
  int64_t ciphertextSize;
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
    Type dataSemanticType = op.getValue().getType();
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getValue().getType());
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

      SmallVector<int64_t> dataSemanticShape;
      if (auto tensorTy = dyn_cast<RankedTensorType>(dataSemanticType)) {
        dataSemanticShape = SmallVector<int64_t>(tensorTy.getShape());
      } else {
        // assumed to be a scalar
        dataSemanticShape = {1};
      }
      iterateIndices(dataSemanticShape, evaluateNextIndex);
      minUnusedTarget = getMinUnusedTarget(permutation);
      minUnusedInput = getMinUnusedInput(permutation);
    }

    auto permuteOp = rewriter.create<tensor_ext::PermuteOp>(
        op.getLoc(), adaptor.getValue(),
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
      setMaterializedAttr({permuteOp, nextOp});
      result = nextOp->getResult(0);
    }

    // TODO(#1591): post-process the layout properly for padding
    rewriter.replaceOp(op, result);
    return success();
  }
};

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
    Value matrix = adaptor.getInputs()[0];
    Value vector = adaptor.getInputs()[1];
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
    auto originalMatrixType =
        cast<RankedTensorType>(op.getInputs()[0].getType());
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
    for (int index = 0; index < numRotations; ++index) {
      // construct vector.rotate(i)
      auto rotationIndexOp = b.create<arith::ConstantIntOp>(index, 64);
      auto rotateOp =
          b.create<tensor_ext::RotateOp>(packedVector, rotationIndexOp);

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

      setMaterializedAttr(
          {rotationIndexOp, rotateOp, extractRowOp, mulOp, addOp});
      accumulator = addOp->getResult(0);
    }
    // Only the last op will need its layout set for later ops to reference.
    setAttributeAssociatedWith(accumulator, kLayoutAttrName, layoutAttr);

    Value summedShifts = accumulator;
    if (originalMatrixType.getShape()[0] == originalMatrixType.getShape()[1]) {
      rewriter.replaceOp(op, summedShifts);
      return;
    }

    // else, necessarily matrixNumRows < matrixNumCols due to the precondition
    // applied earlier. This is the post-processing partial-rotate-and-reduce
    // step required for squat-diagonal packing.
    int64_t matrixNumRows = packedMatrixType.getShape()[0];
    int64_t matrixNumCols = packedMatrixType.getShape()[1];

    int64_t numShifts = (int64_t)(log2(matrixNumCols) - log2(matrixNumRows));
    int64_t shift = matrixNumCols / 2;

    for (int64_t i = 0; i < numShifts; ++i) {
      auto shiftAmountOp = b.create<arith::ConstantIntOp>(shift, 64);
      auto rotateOp =
          b.create<tensor_ext::RotateOp>(summedShifts, shiftAmountOp);
      auto *addOp = b.create(OperationState(
          op->getLoc(), addOpName, {summedShifts, rotateOp.getResult()},
          {rotateOp.getResult().getType()}));
      setMaterializedAttr({shiftAmountOp, rotateOp, addOp});
      setAttributeAssociatedWith(addOp->getResult(0), kLayoutAttrName,
                                 layoutAttr);
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

    setMaterializedAttr({zeroOp, oneOp, createMaskOp, applyMaskOp});
    applyMaskOp->setAttr(kLayoutAttrName, layoutAttr);

    rewriter.replaceOp(op, applyMaskOp);
  }

  LogicalResult matchAndRewrite(
      linalg::MatvecOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    Value matrix = adaptor.getInputs()[0];
    Value vector = adaptor.getInputs()[1];
    LayoutAttr vectorLayout = getLayoutAttr(vector);
    LayoutAttr matrixLayout = getLayoutAttr(matrix);

    if (!matrixLayout)
      return op.emitError() << "missing layout attribute for matrix";

    if (!vectorLayout)
      return op.emitError() << "missing layout attribute for vector";

    if (supportsHaleviShoup(op, adaptor)) {
      haleviShoupKernel(op, adaptor, rewriter);
      return success();
    }

    // TODO(#1589): implement row-major naive matvec kernel
    return op.emitError() << "unsupported layout for matrix in matvec: "
                          << matrixLayout;
  }
};

Value makeMask(ContextAwareConversionPatternRewriter &rewriter, Location loc,
               Value index, RankedTensorType ciphertextSemanticType) {
  // The ciphertext tensor is a 1D tensor, so the applyOp's result is a
  // single value we can use to build a mask.
  // A tensor of zeros
  auto maskHolder = rewriter.create<arith::ConstantOp>(
      loc, ciphertextSemanticType,
      rewriter.getZeroAttr(ciphertextSemanticType));
  // A scalar 1
  auto one = rewriter.create<arith::ConstantOp>(
      loc, ciphertextSemanticType.getElementType(),
      rewriter.getOneAttr(ciphertextSemanticType.getElementType()));
  // insert 1 into the right index
  auto mask = rewriter.create<tensor::InsertOp>(loc, one, maskHolder, index);
  setMaterializedAttr({maskHolder, one, mask});
  return mask.getResult();
}

Value makeInverseMask(ContextAwareConversionPatternRewriter &rewriter,
                      Location loc, Value index,
                      RankedTensorType ciphertextSemanticType) {
  // The ciphertext tensor is a 1D tensor, so the applyOp's result is a
  // single value we can use to build a mask.
  // A tensor of ones
  auto maskHolder = rewriter.create<arith::ConstantOp>(
      loc, ciphertextSemanticType, rewriter.getOneAttr(ciphertextSemanticType));
  // A scalar 0
  auto one = rewriter.create<arith::ConstantOp>(
      loc, ciphertextSemanticType.getElementType(),
      rewriter.getZeroAttr(ciphertextSemanticType.getElementType()));
  // insert 0 into the right index
  auto mask = rewriter.create<tensor::InsertOp>(loc, one, maskHolder, index);
  setMaterializedAttr({maskHolder, one, mask});
  return mask.getResult();
}

class ConvertTensorExtract
    : public ContextAwareOpConversionPattern<tensor::ExtractOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::ExtractOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    // This is not a good op to have a kernel for, but we have it for
    // completeness.
    //
    // The kernel is implemented by masking the tensor at the given index,
    // rotating it to the first position, and then (depending on the
    // replication attribute) replicating it throughout the rest of the tensor.

    FailureOr<Attribute> tensorLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getTensor());
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    LayoutAttr tensorLayout = cast<LayoutAttr>(tensorLayoutResult.value());
    FailureOr<Attribute> resultLayoutResult =
        getTypeConverter()->getContextualAttr(op.getResult());
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    // TODO(#1692) properly handle result layout alignment
    LayoutAttr resultLayout = cast<LayoutAttr>(resultLayoutResult.value());

    // The indices to extract must be materialized via the layout mapping, which
    // corresponds to inserting an affine.apply.
    if (adaptor.getIndices().size() != tensorLayout.getMap().getNumDims()) {
      std::string mapStr;
      llvm::raw_string_ostream os(mapStr);
      tensorLayout.getMap().print(os);
      return op.emitError()
             << "mismatching number of indices (" << adaptor.getIndices().size()
             << ") for map " << mapStr;
    }
    auto applyOp = rewriter.create<affine::AffineApplyOp>(
        op.getLoc(), tensorLayout.getMap(), adaptor.getIndices());

    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getTensor().getType());

    // TODO(#1542): support multi-packed ciphertexts
    if (ciphertextSemanticType.getRank() != 1) {
      return op.emitError()
             << "Does not support packing into multiple ciphertexts yet";
    }

    // The ciphertext tensor is a 1D tensor, so the applyOp's result is a
    // single value we can use to build a mask.
    // A tensor of zeros
    Value mask = makeMask(rewriter, op.getLoc(), applyOp.getResult(),
                          ciphertextSemanticType);

    // multiply the mask by the converted value
    StringRef mulOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.muli"
            : "arith.mulf";
    Operation *mulOp = rewriter.create(
        OperationState(op->getLoc(), mulOpName, {mask, adaptor.getTensor()},
                       {ciphertextSemanticType}));

    // Rotate left to the first position
    auto rotateOp = rewriter.create<tensor_ext::RotateOp>(
        op.getLoc(), mulOp->getResult(0), applyOp.getResult());
    Operation *result = rotateOp;

    // TODO(#1662): improve scalar layout materialization

    setMaterializedAttr({applyOp, mulOp, rotateOp});
    setAttributeAssociatedWith(result->getResult(0), kLayoutAttrName,
                               resultLayout);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertTensorInsert
    : public ContextAwareOpConversionPattern<tensor::InsertOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::InsertOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter &rewriter) const final {
    // This is not a good op to have a kernel for, but we have it for
    // completeness.
    //
    // The kernel is implemented by masking the input at index 0, rotating it
    // to the needed (materialized) index of the dest ciphertext, masking a
    // zero in the dest at the same index, and then adding the two masked
    // tensors.

    FailureOr<Attribute> tensorLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getDest());
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    LayoutAttr tensorLayout = cast<LayoutAttr>(tensorLayoutResult.value());
    FailureOr<Attribute> resultLayoutResult =
        getTypeConverter()->getContextualAttr(op.getResult());
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    LayoutAttr resultLayout = cast<LayoutAttr>(resultLayoutResult.value());

    // The indices at which to insert must be materialized via the layout
    // mapping, which corresponds to inserting an affine.apply.
    auto applyOp = rewriter.create<affine::AffineApplyOp>(
        op.getLoc(), tensorLayout.getMap(), adaptor.getIndices());

    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getDest().getType());

    // TODO(#1542): support multi-packed ciphertexts
    if (ciphertextSemanticType.getRank() != 1) {
      return op.emitError()
             << "Does not support packing into multiple ciphertexts yet";
    }

    StringRef mulOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.muli"
            : "arith.mulf";
    StringRef addOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.addi"
            : "arith.addf";

    // TODO(#1662): support more sophisticated scalar layouts
    //
    // The scalar to insert is materialized to a tensor like
    //
    //   [v, v, v, ..., v]
    //
    // Mask the materialized tensor for the scalar to insert at index zero
    //
    //   [v, 0, 0, ..., 0]
    //
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value mask = makeMask(rewriter, op.getLoc(), zero.getResult(),
                          ciphertextSemanticType);
    Operation *scalarMul = rewriter.create(
        OperationState(op->getLoc(), mulOpName, {mask, adaptor.getScalar()},
                       {ciphertextSemanticType}));

    // Rotate to the (materialized) index to insert
    //
    //   [0, ..., 0, v, 0, ..., 0]
    //
    auto rotateOp = rewriter.create<tensor_ext::RotateOp>(
        op.getLoc(), scalarMul->getResult(0), applyOp.getResult());

    // Inverse-mask the destination tensor so there's a zero at the target
    // value
    //
    //   [a1, a2, ..., an] --> [a1, ..., a_{k-1}, 0, a_{k+1}, ..., an]
    //
    Value inverseMask = makeInverseMask(
        rewriter, op.getLoc(), applyOp.getResult(), ciphertextSemanticType);
    Operation *destMul = rewriter.create(OperationState(
        op->getLoc(), mulOpName, {inverseMask, adaptor.getDest()},
        {ciphertextSemanticType}));

    // Add the two masked tensors together
    // value
    //
    //     [a1, ..., a_{k-1}, 0, a_{k+1}, ..., an]
    //   + [ 0, ...,       0, 0,       v, ...,  0]
    //
    Operation *finalAdd = rewriter.create(
        OperationState(op->getLoc(), addOpName,
                       {scalarMul->getResult(0), destMul->getResult(0)},
                       {ciphertextSemanticType}));
    Value result = finalAdd->getResult(0);

    setMaterializedAttr(
        {applyOp, zero, scalarMul, rotateOp, destMul, finalAdd});
    setAttributeAssociatedWith(result, kLayoutAttrName, resultLayout);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertToCiphertextSemantics
    : impl::ConvertToCiphertextSemanticsBase<ConvertToCiphertextSemantics> {
  using ConvertToCiphertextSemanticsBase::ConvertToCiphertextSemanticsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    int64_t ctSize = ciphertextSize;
    LayoutMaterializationTypeConverter typeConverter =
        LayoutMaterializationTypeConverter(ctSize);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isa<ModuleOp>(op) || hasMaterializedAttr(op);
    });

    patterns.add<ConvertFunc, ConvertGeneric,
                 // tensor_ext ops
                 ConvertConvertLayout,
                 // linalg ops
                 ConvertLinalgReduce, ConvertLinalgMatvec,
                 // tensor ops
                 ConvertTensorExtract, ConvertTensorInsert,
                 // default
                 ConvertAnyAddingMaterializedAttr>(typeConverter, context);
    patterns.add<ConvertAssignLayout>(typeConverter, context, ciphertextSize);

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
