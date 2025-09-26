#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "lib/Transforms/DropUnitDims/DropUnitDims.h"
#include "lib/Utils/AffineMapUtils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/TransformUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
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
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

namespace {
using kernel::ArithmeticDagNode;
using kernel::implementHaleviShoup;
using kernel::IRMaterializingVisitor;
using kernel::SSAValue;
using presburger::IntegerRelation;
using presburger::VarKind;
using tensor_ext::LayoutAttr;
using tensor_ext::NewLayoutAttr;

auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
auto& kMaterializedAttrName = "tensor_ext.layout_materialized";
auto& kOriginalTypeAttrName =
    tensor_ext::TensorExtDialect::kOriginalTypeAttrName;

}  // namespace

// An unset value of a permutation as it's being built up.
static constexpr int kUnset = -1;

#define GEN_PASS_DEF_CONVERTTOCIPHERTEXTSEMANTICS
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h.inc"

std::optional<Type> convertTypeWithLayout(Type type, LayoutAttr attr,
                                          int64_t ciphertextSize) {
  return llvm::TypeSwitch<Type, std::optional<Type>>(type)
      .Case<secret::SecretType>([&](auto type) -> std::optional<Type> {
        FailureOr<Type> convertedInnerType;
        auto innerType = type.getValueType();

        if (auto rankedTensorType = dyn_cast<RankedTensorType>(innerType)) {
          convertedInnerType =
              materializeLayout(rankedTensorType, attr, ciphertextSize);
        } else {
          convertedInnerType =
              materializeScalarLayout(innerType, attr, ciphertextSize);
        }

        if (failed(convertedInnerType)) return std::nullopt;
        return secret::SecretType::get(convertedInnerType.value());
      })
      .Case<RankedTensorType>([&](auto type) -> std::optional<Type> {
        return materializeLayout(type, attr, ciphertextSize);
      })
      .Case<IntegerType, FloatType, IndexType>(
          [&](auto type) -> std::optional<Type> {
            return materializeScalarLayout(type, attr, ciphertextSize);
          })
      .Default([&](auto type) -> std::optional<Type> { return std::nullopt; });
}

std::optional<Type> convertTypeWithLayout(Type type, NewLayoutAttr attr,
                                          int64_t ciphertextSize) {
  return llvm::TypeSwitch<Type, std::optional<Type>>(type)
      .Case<secret::SecretType>([&](auto type) -> std::optional<Type> {
        auto innerType = type.getValueType();

        FailureOr<Type> convertedInnerType = materializeNewLayout(
            getElementTypeOrSelf(innerType), attr, ciphertextSize);
        if (failed(convertedInnerType)) return std::nullopt;
        return secret::SecretType::get(convertedInnerType.value());
      })
      .Case<RankedTensorType, IntegerType, FloatType, IndexType>(
          [&](auto type) -> std::optional<Type> {
            return materializeNewLayout(getElementTypeOrSelf(type), attr,
                                        ciphertextSize);
          })
      .Default([&](auto type) -> std::optional<Type> { return std::nullopt; });
}

std::optional<Type> convertTypeWithAttr(Type type, Attribute attr,
                                        int64_t ciphertextSize) {
  return llvm::TypeSwitch<Attribute, std::optional<Type>>(attr)
      .Case<LayoutAttr>([&](auto attr) -> std::optional<Type> {
        return convertTypeWithLayout(type, attr, ciphertextSize);
      })
      .Case<NewLayoutAttr>([&](auto attr) -> std::optional<Type> {
        return convertTypeWithLayout(type, attr, ciphertextSize);
      })
      .Default([&](auto attr) -> std::optional<Type> { return std::nullopt; });
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
struct LayoutMaterializationTypeConverter : public TypeConverter {
 public:
  LayoutMaterializationTypeConverter(int ciphertextSize)
      : TypeConverter(), ciphertextSize(ciphertextSize) {
    addConversion([&](Value value) -> std::optional<Type> {
      FailureOr<Attribute> layoutAttr =
          findAttributeAssociatedWith(value, kLayoutAttrName);
      if (failed(layoutAttr)) return std::nullopt;
      return convertTypeWithAttr(value.getType(), layoutAttr.value(),
                                 getCiphertextSize());
    });
  }

  int getCiphertextSize() const { return ciphertextSize; }

 private:
  int ciphertextSize;
};

bool hasMaterializedAttr(Operation* op) {
  return op->hasAttr(kMaterializedAttrName);
}

void setMaterializedAttr(Operation* op) {
  op->setAttr(kMaterializedAttrName, UnitAttr::get(op->getContext()));
}

void setMaterializedAttr(ArrayRef<Operation*> ops) {
  for (auto* op : ops) {
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
  ConvertFunc(const TypeConverter& converter, MLIRContext* context,
              int64_t ciphertextSize)
      : ContextAwareFuncConversion(
            converter, context, std::string(kLayoutAttrName),
            [&](Type type, Attribute attr) {
              return convertTypeWithAttr(type, attr, ciphertextSize);
            }) {}

  LogicalResult finalizeFuncOpModification(
      func::FuncOp op, ArrayRef<Type> oldArgTypes,
      ArrayRef<Type> oldResultTypes, PatternRewriter& rewriter) const override {
    // Replace layout arg attrs with secret.original_type arg attrs. This is
    // necessary so that later encoding/decoding functions can know what the
    // original type of the tensor was and how it was encoded.
    rewriter.modifyOpInPlace(op, [&] {
      setMaterializedAttr(op);
      for (int i = 0; i < op.getNumArguments(); ++i) {
        auto layoutAttr = op.getArgAttr(i, kLayoutAttrName);
        if (!layoutAttr ||
            !(isa<LayoutAttr>(layoutAttr) || isa<NewLayoutAttr>(layoutAttr))) {
          continue;
        }

        op.setArgAttr(i, kOriginalTypeAttrName,
                      tensor_ext::OriginalTypeAttr::get(
                          getContext(), maybeExtractSecretType(oldArgTypes[i]),
                          layoutAttr));
      }

      for (int i = 0; i < op.getNumResults(); ++i) {
        auto layoutAttr = op.getResultAttr(i, kLayoutAttrName);
        if (!layoutAttr ||
            !(isa<LayoutAttr>(layoutAttr) || isa<NewLayoutAttr>(layoutAttr))) {
          continue;
        }

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

struct ConvertGeneric : public ConvertAny<secret::GenericOp> {
 public:
  ConvertGeneric(const TypeConverter& converter, MLIRContext* context)
      : ConvertAny(converter, context) {
    setDebugName("ConvertGeneric");
  }

  LogicalResult finalizeOpModification(
      secret::GenericOp op,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] { setMaterializedAttr(op); });
    return success();
  };
};

// Convert an op generically, marking it as materialized. Lowest priority
// because it is only meant to handle ops that don't have special
// materialization rules.
struct ConvertAnyAddingMaterializedAttr : public ConvertAny<> {
  ConvertAnyAddingMaterializedAttr(const TypeConverter& converter,
                                   MLIRContext* context)
      : ConvertAny(converter, context, /*benefit=*/0) {
    setDebugName("ConvertAnyAddingMaterializedAttr");
  }

  LogicalResult finalizeOpModification(
      Operation* op, ConversionPatternRewriter& rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] { setMaterializedAttr(op); });
    return success();
  };
};

class ConvertAssignLayout
    : public OpConversionPattern<tensor_ext::AssignLayoutOp> {
 public:
  ConvertAssignLayout(const TypeConverter& typeConverter,
                      mlir::MLIRContext* context, int64_t ciphertextSize)
      : OpConversionPattern<tensor_ext::AssignLayoutOp>(typeConverter, context),
        ciphertextSize(ciphertextSize) {}

  LogicalResult matchAndRewrite(
      tensor_ext::AssignLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto res =
        implementAssignLayout(op, ciphertextSize, b, [&](Operation* createdOp) {
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

class ConvertConvertLayout
    : public OpConversionPattern<tensor_ext::ConvertLayoutOp> {
 public:
  using OpConversionPattern<tensor_ext::ConvertLayoutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor_ext::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Type dataSemanticType = op.getValue().getType();
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getValue().getType());
    LLVM_DEBUG(llvm::dbgs()
               << "ConvertConvertLayout: dataSemanticType=" << dataSemanticType
               << ", ciphertextSemanticType=" << ciphertextSemanticType
               << "\n");

    // TODO(#2047): Support new layout attr
    LayoutAttr fromLayout = dyn_cast<LayoutAttr>(op.getFromLayout());
    LayoutAttr toLayout = dyn_cast<LayoutAttr>(op.getToLayout());
    if (!fromLayout || !toLayout) {
      return failure();  // soon to be deleted
    }

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
          [&](const std::vector<int64_t>& indices) {
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

    auto permuteOp =
        tensor_ext::PermuteOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                                      rewriter.getI64TensorAttr(permutation));
    permuteOp->setAttr(kLayoutAttrName, op->getAttr(kLayoutAttrName));
    setMaterializedAttr(permuteOp);
    rewriter.replaceOp(op, permuteOp);
    return success();
  };
};

class ConvertConvertLayoutNewLayout
    : public OpConversionPattern<tensor_ext::ConvertLayoutOp> {
 public:
  using OpConversionPattern<tensor_ext::ConvertLayoutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor_ext::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    NewLayoutAttr fromLayout = dyn_cast<NewLayoutAttr>(op.getFromLayout());
    NewLayoutAttr toLayout = dyn_cast<NewLayoutAttr>(op.getToLayout());
    if (!fromLayout || !toLayout) {
      return failure();
    }

    // This is persisted as an operation rather than lowered eagerly to a shift
    // network so as to allow VosVosErkinShiftNetworks to cache its work across
    // multiple instances of the same conversion.
    std::shared_ptr<IntegerRelation> toLayoutClone =
        toLayout.getIntegerRelation().clone();
    toLayoutClone->inverse();
    toLayoutClone->compose(fromLayout.getIntegerRelation());
    NewLayoutAttr combinedLayout =
        NewLayoutAttr::getFromIntegerRelation(getContext(), *toLayoutClone);

    auto permuteOp = tensor_ext::PermuteOp::create(
        rewriter, op.getLoc(), adaptor.getValue(), combinedLayout);

    setMaterializedAttr(permuteOp);
    setAttributeAssociatedWith(permuteOp, kLayoutAttrName, toLayout);
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
  for (const auto& [input, output] :
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

class ConvertLinalgReduce : public OpConversionPattern<linalg::ReduceOp> {
 public:
  using OpConversionPattern<linalg::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Ensure the reduction op is single addition or multiplication, otherwise
    // there is no kernel.
    Block* body = op.getBlock();
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

    Operation* innerOp = &body->getOperations().front();
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
        findAttributeAssociatedWith(adaptor.getInputs()[0], kLayoutAttrName);
    if (failed(layoutFetchResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    // TODO(#2047): Support new layout attr
    LayoutAttr layout = dyn_cast<LayoutAttr>(layoutFetchResult.value());
    if (!layout) {
      return op.emitError() << "expected LayoutAttr on linalg.reduce";
    }

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
          [&](const std::vector<int64_t>& indices) {
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

      auto permuteOp = tensor_ext::PermuteOp::create(
          b, input, b.getI64TensorAttr(permutation));

      SmallVector<Value> operands = {result, permuteOp};
      SmallVector<Type> newResultTypes = {permuteOp.getType()};
      Operation* nextOp = rewriter.create(
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

struct ConvertLinalgMatvec : public OpConversionPattern<linalg::MatvecOp> {
 public:
  using OpConversionPattern<linalg::MatvecOp>::OpConversionPattern;

  LayoutAttr getLayoutAttr(Value value) const {
    auto layoutLookup = findAttributeAssociatedWith(value, kLayoutAttrName);
    if (failed(layoutLookup)) {
      return nullptr;
    }
    return dyn_cast<LayoutAttr>(layoutLookup.value());
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

  void haleviShoupKernel(linalg::MatvecOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter& rewriter) const {
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
      auto rotationIndexOp = arith::ConstantIntOp::create(b, index, 64);
      auto rotateOp =
          tensor_ext::RotateOp::create(b, packedVector, rotationIndexOp);

      // get the corresponding element of the ciphertext tensor,
      // which is row i
      SmallVector<OpFoldResult> offsets = {b.getIndexAttr(index),
                                           b.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {
          b.getIndexAttr(1), b.getIndexAttr(packedMatrixType.getShape()[1])};
      SmallVector<OpFoldResult> strides = {b.getIndexAttr(1),
                                           b.getIndexAttr(1)};
      auto extractRowOp = tensor::ExtractSliceOp::create(
          b, packedVectorType, packedMatrix, offsets, sizes, strides);

      Operation* mulOp = b.create(
          OperationState(op->getLoc(), mulOpName,
                         {rotateOp.getResult(), extractRowOp.getResult()},
                         {packedVectorType}));
      Operation* addOp = b.create(OperationState(
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
      auto shiftAmountOp = arith::ConstantIntOp::create(b, shift, 64);
      auto rotateOp =
          tensor_ext::RotateOp::create(b, summedShifts, shiftAmountOp);
      auto* addOp = b.create(OperationState(
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
    auto zeroOp = arith::ConstantOp::create(b, result.getType(), padAttr);

    // insert a slice of 1's in the first n of the zeros tensor to make a mask
    SmallVector<int64_t> prefixShape = {matrixNumRows};
    RankedTensorType prefixType =
        RankedTensorType::get(prefixShape, elementType);
    auto oneOp =
        arith::ConstantOp::create(b, prefixType, b.getOneAttr(prefixType));
    auto createMaskOp = tensor::InsertSliceOp::create(
        b, oneOp, zeroOp, ArrayRef<Value>{}, ArrayRef<Value>{},
        ArrayRef<Value>{},
        /*offsets=*/ArrayRef<int64_t>{0},
        /*sizes=*/ArrayRef{matrixNumRows}, /*strides=*/ArrayRef<int64_t>{1});
    auto* applyMaskOp = b.create(OperationState(op->getLoc(), mulOpName,
                                                {summedShifts, createMaskOp},
                                                {summedShifts.getType()}));

    setMaterializedAttr({zeroOp, oneOp, createMaskOp, applyMaskOp});
    applyMaskOp->setAttr(kLayoutAttrName, layoutAttr);

    rewriter.replaceOp(op, applyMaskOp);
  }

  LogicalResult matchAndRewrite(
      linalg::MatvecOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Value matrix = adaptor.getInputs()[0];
    Value vector = adaptor.getInputs()[1];
    LayoutAttr vectorLayout = getLayoutAttr(vector);
    LayoutAttr matrixLayout = getLayoutAttr(matrix);

    if (!matrixLayout || !vectorLayout)
      return rewriter.notifyMatchFailure(
          op, "missing layout attribute matrix and vector");

    if (supportsHaleviShoup(op, adaptor)) {
      haleviShoupKernel(op, adaptor, rewriter);
      return success();
    }

    // TODO(#1589): implement row-major naive matvec kernel
    return op.emitError() << "unsupported layout for matrix in matvec: "
                          << matrixLayout;
  }
};

struct ConvertLinalgMatvecNewLayout
    : public OpConversionPattern<linalg::MatvecOp> {
 public:
  using OpConversionPattern<linalg::MatvecOp>::OpConversionPattern;

  ConvertLinalgMatvecNewLayout(const TypeConverter& contextAwareTypeConverter,
                               MLIRContext* context)
      : OpConversionPattern(contextAwareTypeConverter, context,
                            /*benefit=*/10) {}

  NewLayoutAttr getNewLayoutAttr(Value value) const {
    auto layoutLookup = findAttributeAssociatedWith(value, kLayoutAttrName);
    if (failed(layoutLookup)) {
      return nullptr;
    }
    return dyn_cast<NewLayoutAttr>(layoutLookup.value());
  }

  bool supportsHaleviShoup(linalg::MatvecOp op, OpAdaptor adaptor) const {
    Value matrix = adaptor.getInputs()[0];
    Value vector = adaptor.getInputs()[1];
    Value originalVector = op.getInputs()[1];
    // auto vectorType = cast<RankedTensorType>(vector.getType());
    auto materializedMatrixType = cast<RankedTensorType>(matrix.getType());
    auto materializedVectorType = cast<RankedTensorType>(vector.getType());

    // If one of these dimensions is not a power of two, then we can't do
    // the Halevi-Shoup or Squat Packing Matrix Multiplication conversion.
    auto dimensions = materializedMatrixType.getShape();
    int64_t numRows = dimensions[0];
    int64_t numCols = dimensions[1];
    if (!isPowerOfTwo(numRows) || !isPowerOfTwo(numCols)) {
      return false;
    }

    NewLayoutAttr matrixLayout = getNewLayoutAttr(matrix);
    NewLayoutAttr vectorLayout = getNewLayoutAttr(vector);
    bool isSquatDiagonal = isRelationSquatDiagonal(
        cast<RankedTensorType>(op.getInputs()[0].getType()), numCols,
        matrixLayout.getIntegerRelation());
    bool isRowMajor =
        isRelationRowMajor(cast<RankedTensorType>(originalVector.getType()),
                           materializedVectorType.getDimSize(1),
                           vectorLayout.getIntegerRelation());

    LLVM_DEBUG(llvm::dbgs()
               << "supportsHaleviShoup: " << "isSquatDiagonal="
               << isSquatDiagonal << " isRowMajor=" << isRowMajor << "\n");

    // TODO(#1578): If the matrix has more rows than columns, what kernel
    // should be used?
    bool dimensionsCompatible = numRows <= numCols;
    return isSquatDiagonal && isRowMajor && dimensionsCompatible;
  }

  void haleviShoupKernel(linalg::MatvecOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "Converting linalg.matvec op with halevi shoup kernel: " << op
               << "\n");

    TypedValue<RankedTensorType> input =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[1]);
    SSAValue vectorLeaf(input);
    TypedValue<RankedTensorType> matrix =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[0]);
    SSAValue matrixLeaf(matrix);

    std::shared_ptr<ArithmeticDagNode<SSAValue>> implementedKernel =
        implementHaleviShoup(
            vectorLeaf, matrixLeaf,
            cast<RankedTensorType>(op.getInputs()[0].getType()).getShape());

    rewriter.setInsertionPointAfter(op);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    IRMaterializingVisitor visitor(b, input.getType());
    Value finalOutput = implementedKernel->visit(visitor);

    auto layoutAttr = cast<NewLayoutAttr>(op->getAttr(kLayoutAttrName));
    auto finalOutputOp = finalOutput.getDefiningOp();
    finalOutputOp->setAttr(kLayoutAttrName, layoutAttr);
    setMaterializedAttr(finalOutputOp);

    // Add the initial accumulator value.
    Value result = adaptor.getOutputs()[0];
    Operation* addBias =
        makeAppropriatelyTypedAddOp(b, op->getLoc(), finalOutput, result);
    addBias->setAttr(kLayoutAttrName, layoutAttr);
    setMaterializedAttr(addBias);
    rewriter.replaceOp(op, addBias);
  }

  LogicalResult matchAndRewrite(
      linalg::MatvecOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Value matrix = adaptor.getInputs()[0];
    Value vector = adaptor.getInputs()[1];
    NewLayoutAttr vectorLayout = getNewLayoutAttr(vector);
    NewLayoutAttr matrixLayout = getNewLayoutAttr(matrix);

    if (!matrixLayout || !vectorLayout)
      return rewriter.notifyMatchFailure(
          op, "missing new layout attribute for matrix and vector");

    if (supportsHaleviShoup(op, adaptor)) {
      haleviShoupKernel(op, adaptor, rewriter);
      return success();
    }

    // TODO(#1589): implement row-major naive matvec kernel
    return op.emitError() << "unsupported layout for matrix in matvec: "
                          << matrixLayout;
  }
};

struct ConvertLinalgConv2D : public OpConversionPattern<linalg::Conv2DOp> {
 public:
  using OpConversionPattern<linalg::Conv2DOp>::OpConversionPattern;

  ConvertLinalgConv2D(const TypeConverter& contextAwareTypeConverter,
                      MLIRContext* context)
      : OpConversionPattern(contextAwareTypeConverter, context,
                            /*benefit=*/10) {}

  NewLayoutAttr getNewLayoutAttr(Value value) const {
    auto layoutLookup = findAttributeAssociatedWith(value, kLayoutAttrName);
    if (failed(layoutLookup)) {
      return nullptr;
    }
    return dyn_cast<NewLayoutAttr>(layoutLookup.value());
  }

  bool supportsExpandedHaleviShoup(linalg::Conv2DOp op,
                                   OpAdaptor adaptor) const {
    Value filter = adaptor.getInputs().back();
    auto materializedFilterType = cast<RankedTensorType>(filter.getType());

    // If one of these dimensions is not a power of two, then we can't do
    // the Halevi-Shoup or Squat Packing Matrix Multiplication conversion.
    auto dimensions = materializedFilterType.getShape();
    int64_t numRows = dimensions[0];
    int64_t numCols = dimensions[1];
    bool isPowerOfTwoDims = isPowerOfTwo(numRows) && isPowerOfTwo(numCols);

    // TODO(#1578): If the matrix has more rows than columns, what kernel
    // should be used?
    bool isMatrixCompatible = numRows <= numCols;

    auto kernelAttr = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    bool isConv2dAsMatvec =
        kernelAttr && kernelAttr.getName() == KernelName::MatvecDiagonal;

    LLVM_DEBUG(llvm::dbgs()
               << "supports expanded conv2d as matvec with halevi-shoup: "
               << "isPowerOfTwoDims=" << isPowerOfTwoDims
               << " isMatrixCompatible=" << isMatrixCompatible
               << " isConv2dAsMatvec=" << isConv2dAsMatvec << "\n");

    return isPowerOfTwoDims && isMatrixCompatible && isConv2dAsMatvec;
  }

  void haleviShoupKernel(linalg::Conv2DOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "Converting linalg.conv2d op with halevi shoup kernel: " << op
               << "\n");

    TypedValue<RankedTensorType> data =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[0]);
    SSAValue vectorLeaf(data);
    TypedValue<RankedTensorType> matrix =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[1]);
    SSAValue matrixLeaf(matrix);

    std::shared_ptr<ArithmeticDagNode<SSAValue>> implementedKernel =
        implementHaleviShoup(
            vectorLeaf, matrixLeaf,
            cast<RankedTensorType>(op.getInputs()[0].getType()).getShape());

    rewriter.setInsertionPointAfter(op);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    IRMaterializingVisitor visitor(b, data.getType());
    Value finalOutput = implementedKernel->visit(visitor);

    auto layoutAttr = cast<NewLayoutAttr>(op->getAttr(kLayoutAttrName));
    auto finalOutputOp = finalOutput.getDefiningOp();
    finalOutputOp->setAttr(kLayoutAttrName, layoutAttr);
    setMaterializedAttr(finalOutputOp);

    // Add the initial accumulator value.
    Value result = adaptor.getOutputs()[0];
    Operation* addBias =
        makeAppropriatelyTypedAddOp(b, op->getLoc(), finalOutput, result);
    addBias->setAttr(kLayoutAttrName, layoutAttr);
    setMaterializedAttr(addBias);
    rewriter.replaceOp(op, addBias);
  }

  LogicalResult matchAndRewrite(
      linalg::Conv2DOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Value data = adaptor.getInputs().front();
    Value filter = adaptor.getInputs().back();
    NewLayoutAttr dataLayout = getNewLayoutAttr(data);
    NewLayoutAttr filterLayout = getNewLayoutAttr(filter);

    if (!dataLayout || !filterLayout)
      return rewriter.notifyMatchFailure(
          op, "missing new layout attribute for data and filter");

    if (supportsExpandedHaleviShoup(op, adaptor)) {
      haleviShoupKernel(op, adaptor, rewriter);
      return success();
    }

    return op.emitError() << "unsupported layout for 2d conv";
  }
};

Value makeMask(ConversionPatternRewriter& rewriter, Location loc, Value index,
               RankedTensorType ciphertextSemanticType) {
  // The ciphertext tensor is a 1D tensor, so the applyOp's result is a
  // single value we can use to build a mask.
  // A tensor of zeros
  auto maskHolder =
      arith::ConstantOp::create(rewriter, loc, ciphertextSemanticType,
                                rewriter.getZeroAttr(ciphertextSemanticType));
  // A scalar 1
  auto one = arith::ConstantOp::create(
      rewriter, loc, ciphertextSemanticType.getElementType(),
      rewriter.getOneAttr(ciphertextSemanticType.getElementType()));
  // insert 1 into the right index
  auto mask = tensor::InsertOp::create(rewriter, loc, one, maskHolder, index);
  setMaterializedAttr({maskHolder, one, mask});
  return mask.getResult();
}

Value makeInverseMask(ConversionPatternRewriter& rewriter, Location loc,
                      Value index, RankedTensorType ciphertextSemanticType) {
  // The ciphertext tensor is a 1D tensor, so the applyOp's result is a
  // single value we can use to build a mask.
  // A tensor of ones
  auto maskHolder =
      arith::ConstantOp::create(rewriter, loc, ciphertextSemanticType,
                                rewriter.getOneAttr(ciphertextSemanticType));
  // A scalar 0
  auto one = arith::ConstantOp::create(
      rewriter, loc, ciphertextSemanticType.getElementType(),
      rewriter.getZeroAttr(ciphertextSemanticType.getElementType()));
  // insert 0 into the right index
  auto mask = tensor::InsertOp::create(rewriter, loc, one, maskHolder, index);
  setMaterializedAttr({maskHolder, one, mask});
  return mask.getResult();
}

class ConvertTensorExtract : public OpConversionPattern<tensor::ExtractOp> {
 public:
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // This is not a good op to have a kernel for, but we have it for
    // completeness.
    //
    // The kernel is implemented by masking the tensor at the given index,
    // rotating it to the first position, and then (depending on the
    // replication attribute) replicating it throughout the rest of the tensor.

    FailureOr<Attribute> tensorLayoutResult =
        findAttributeAssociatedWith(adaptor.getTensor(), kLayoutAttrName);
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    FailureOr<Attribute> resultLayoutResult =
        findAttributeAssociatedWith(op.getResult(), kLayoutAttrName);
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }

    LayoutAttr tensorLayout = dyn_cast<LayoutAttr>(tensorLayoutResult.value());
    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(resultLayoutResult.value());

    if (!tensorLayout || !resultLayout) {
      return failure();  // code will be deleted soon, no need to give message
    }

    // TODO(#1692) properly handle result layout alignment

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
    auto applyOp = affine::AffineApplyOp::create(
        rewriter, op.getLoc(), tensorLayout.getMap(), adaptor.getIndices());

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
    Operation* mulOp = rewriter.create(
        OperationState(op->getLoc(), mulOpName, {mask, adaptor.getTensor()},
                       {ciphertextSemanticType}));

    // Rotate left to the first position
    auto rotateOp = tensor_ext::RotateOp::create(
        rewriter, op.getLoc(), mulOp->getResult(0), applyOp.getResult());
    Operation* result = rotateOp;

    // TODO(#1662): improve scalar layout materialization

    setMaterializedAttr({applyOp, mulOp, rotateOp});
    setAttributeAssociatedWith(result->getResult(0), kLayoutAttrName,
                               resultLayout);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertTensorInsert : public OpConversionPattern<tensor::InsertOp> {
 public:
  using OpConversionPattern<tensor::InsertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // This is not a good op to have a kernel for, but we have it for
    // completeness.
    //
    // The kernel is implemented by masking the input at index 0, rotating it
    // to the needed (materialized) index of the dest ciphertext, masking a
    // zero in the dest at the same index, and then adding the two masked
    // tensors.

    FailureOr<Attribute> tensorLayoutResult =
        findAttributeAssociatedWith(adaptor.getDest(), kLayoutAttrName);
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    FailureOr<Attribute> resultLayoutResult =
        findAttributeAssociatedWith(op.getResult(), kLayoutAttrName);
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }

    LayoutAttr tensorLayout = dyn_cast<LayoutAttr>(tensorLayoutResult.value());
    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(resultLayoutResult.value());

    if (!tensorLayout || !resultLayout) {
      return failure();  // code will be deleted soon, no need to give message
    }

    // The indices at which to insert must be materialized via the layout
    // mapping, which corresponds to inserting an affine.apply.
    auto applyOp = affine::AffineApplyOp::create(
        rewriter, op.getLoc(), tensorLayout.getMap(), adaptor.getIndices());

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
    auto zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    Value mask = makeMask(rewriter, op.getLoc(), zero.getResult(),
                          ciphertextSemanticType);
    Operation* scalarMul = rewriter.create(
        OperationState(op->getLoc(), mulOpName, {mask, adaptor.getScalar()},
                       {ciphertextSemanticType}));

    // Rotate to the (materialized) index to insert
    //
    //   [0, ..., 0, v, 0, ..., 0]
    //
    auto rotateOp = tensor_ext::RotateOp::create(
        rewriter, op.getLoc(), scalarMul->getResult(0), applyOp.getResult());

    // Inverse-mask the destination tensor so there's a zero at the target
    // value
    //
    //   [a1, a2, ..., an] --> [a1, ..., a_{k-1}, 0, a_{k+1}, ..., an]
    //
    Value inverseMask = makeInverseMask(
        rewriter, op.getLoc(), applyOp.getResult(), ciphertextSemanticType);
    Operation* destMul = rewriter.create(OperationState(
        op->getLoc(), mulOpName, {inverseMask, adaptor.getDest()},
        {ciphertextSemanticType}));

    // Add the two masked tensors together
    // value
    //
    //     [a1, ..., a_{k-1}, 0, a_{k+1}, ..., an]
    //   + [ 0, ...,       0, 0,       v, ...,  0]
    //
    Operation* finalAdd = rewriter.create(
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

// Generate IR that loops over the (ciphertext, slot) pairs of the integer
// relation, and at each iteration checks if the corresponding domain index
// tuple matches the dynamicIndices given. If so, it calls thenBuilder
// (forwarded to the then block of an scf.if) with the calculated slot indices
// and iter args, and the return values from thenBuilder are yielded as iter
// args to the next loop iteration.
//
// This is used in some kernels in this file to convert insertion accesses of
// data-semantic tensors into plaintext masks on the corresponding slots of
// ciphertext-semantic tensors.
static FailureOr<SmallVector<Value>> generateLoopWithDynamicIndexCheck(
    ImplicitLocOpBuilder& b, const IntegerRelation& relation,
    ValueRange iterInits, ValueRange dynamicIndices,
    function_ref<SmallVector<Value>(OpBuilder&, Location, ValueRange,
                                    ValueRange)>
        thenBuilder) {
  MLIRLoopNestGenerator generator(b);
  auto loop = generator.generateForLoop(
      relation, iterInits,
      [&](OpBuilder& builder, Location loc, ValueRange exprs,
          ValueRange iterArgs) {
        SmallVector<Value> indices;
        for (int i = relation.getVarKindOffset(VarKind::Range);
             i < relation.getVarKindOffset(VarKind::Range) +
                     relation.getNumRangeVars();
             ++i) {
          indices.push_back(exprs[i]);
        }

        // If statement to check that the current set of exprs matches
        // the dynamic insertion indices.
        Value exprsMatchIndices;

        // conjunction over all exprs[i] == indices[i] where i is a domain
        // variable.
        int domainStart = relation.getVarKindOffset(VarKind::Domain);
        for (int i = domainStart; i < domainStart + relation.getNumDomainVars();
             ++i) {
          auto eqOp =
              arith::CmpIOp::create(b, arith::CmpIPredicate::eq, exprs[i],
                                    dynamicIndices[i - domainStart]);
          setMaterializedAttr(eqOp);

          if (i == domainStart) {
            exprsMatchIndices = eqOp.getResult();
          } else {
            auto andOp =
                arith::AndIOp::create(b, exprsMatchIndices, eqOp.getResult());
            exprsMatchIndices = andOp.getResult();
            setMaterializedAttr(andOp);
          }
        }

        auto ifOp = scf::IfOp::create(
            b, exprsMatchIndices,
            /*thenBuilder=*/
            [&](OpBuilder& builder, Location loc) {
              SmallVector<Value> results =
                  thenBuilder(builder, loc, indices, iterArgs);
              scf::YieldOp::create(b, loc, results);
            },
            /*elseBuilder=*/
            [&](OpBuilder& builder, Location loc) {
              // No-op, just scf.yield the current
              // iter args
              scf::YieldOp::create(b, loc, iterArgs);
            });
        setMaterializedAttr(ifOp);

        // Here the scalarMaskValue can be an actual scalar (in the case of
        // a cleartext scalar) or a 1 (in the case of a secret scalar, so
        // that the resulting mask requires an extra mul).
        return scf::ValueVector(ifOp.getResults());
      });
  if (failed(loop)) {
    return failure();
  }

  SmallVector<Value> results = loop.value().getResults();
  return results;
}

class ConvertTensorExtractNewLayout
    : public OpConversionPattern<tensor::ExtractOp> {
 public:
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

  struct MaskResult {
    Value mask;
    IntegerRelation maskLayout;
  };

  // A helper to create a mask for the case of tensor.extract from a ciphertext
  // when the extraction indices are statically known.
  MaskResult createMaskFromStaticIndices(NewLayoutAttr tensorLayout,
                                         ArrayRef<int64_t> staticIndices,
                                         OpAdaptor adaptor,
                                         ImplicitLocOpBuilder& b) const {
    IntegerRelation tensorRel = tensorLayout.getIntegerRelation();
    IntegerRelation relWithFixedDomain =
        fixDomainVars(tensorRel, staticIndices);
    relWithFixedDomain.projectOut(tensorRel.getVarKindOffset(VarKind::Domain),
                                  tensorRel.getNumDomainVars());

    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getTensor().getType());
    arith::ConstantOp zeroTensorOp = arith::ConstantOp::create(
        b, ciphertextSemanticType, b.getZeroAttr(ciphertextSemanticType));
    arith::ConstantOp oneScalarOp = arith::ConstantOp::create(
        b, ciphertextSemanticType.getElementType(),
        b.getOneAttr(ciphertextSemanticType.getElementType()));

    // TODO(#2216): use a smarter mask via the desired scalar result layout
    std::vector<int64_t> targetSlot = anyRangePoint(relWithFixedDomain);
    SmallVector<Value> indexCsts;
    for (int64_t v : targetSlot) {
      auto cst = arith::ConstantIndexOp::create(b, v);
      indexCsts.push_back(cst);
      setMaterializedAttr(cst);
    }

    auto insertIntoMask =
        tensor::InsertOp::create(b, oneScalarOp, zeroTensorOp, indexCsts);
    setMaterializedAttr({insertIntoMask, zeroTensorOp, oneScalarOp});

    // Since we only insert a 1 in a single range index, the actual layout
    // here requires the additional constraint that we fix the range variables
    // to that chosen point.
    IntegerRelation resultRelation =
        fixRangeVars(relWithFixedDomain, targetSlot);
    return MaskResult{insertIntoMask.getResult(), resultRelation};
  }

  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Mask the tensor at the given index, rotate it to the first position, and
    // then ensure its layout matches the expected output layout with a
    // layout_conversion op.
    FailureOr<Attribute> tensorLayoutResult =
        findAttributeAssociatedWith(adaptor.getTensor(), kLayoutAttrName);
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    FailureOr<Attribute> resultLayoutResult =
        findAttributeAssociatedWith(op.getResult(), kLayoutAttrName);
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }

    NewLayoutAttr tensorLayout =
        cast<NewLayoutAttr>(tensorLayoutResult.value());
    NewLayoutAttr resultLayout =
        cast<NewLayoutAttr>(resultLayoutResult.value());
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getTensor().getType());

    SmallVector<Value> indices(op.getIndices());
    auto staticIndicesResult = getConstantIntValues(getAsOpFoldResult(indices));
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (!staticIndicesResult.has_value()) {
      // mask = createMaskFromDynamicIndices(tensorLayout, adaptor, b);
      //
      // The problem is that, although we could construct the right mask
      // dynamically, then we need to figure out how to convert the masked
      // ciphertext from its natural layout (zeros in most slots) to the
      // appropriate scalar layout (perhaps replicated everywhere, depending on
      // what layout-optimization picks for the result of this op). This
      // requires, at the very least, finding the index of the first nonzero
      // slot (which we can determine from the mask using a loop, yikes), and
      // then rotating the masked ciphertext to put that first nonzero slot in
      // slot 0, and masking out the rest of the slots so we statically know the
      // layout.
      //
      // As a consequence, we need to perform a rotation by a dynamic index. And
      // it's not clear how we could determine which rotation keys to generate
      // to support that.
      //
      // Punting until we have a strong need for this use case.
      return failure();
    }

    MaskResult maskResult = createMaskFromStaticIndices(
        tensorLayout, staticIndicesResult.value(), adaptor, b);
    Value mask = maskResult.mask;

    StringRef mulOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.muli"
            : "arith.mulf";
    Operation* mulOp = rewriter.create(
        OperationState(op->getLoc(), mulOpName, {mask, adaptor.getTensor()},
                       {ciphertextSemanticType}));

    auto convertLayoutOp = tensor_ext::ConvertLayoutOp::create(
        b, mulOp->getResult(0),
        NewLayoutAttr::getFromIntegerRelation(op.getContext(),
                                              maskResult.maskLayout),
        resultLayout);

    // Intentionally don't materialize convert_layout so it will be processed
    // by a subsequent pattern.
    setMaterializedAttr(mulOp);
    setAttributeAssociatedWith(convertLayoutOp.getResult(), kLayoutAttrName,
                               resultLayout);
    rewriter.replaceOp(op, convertLayoutOp);

    convertLayoutOp->getParentOfType<func::FuncOp>().dump();
    return success();
  }
};

class ConvertTensorInsertNewLayout
    : public OpConversionPattern<tensor::InsertOp> {
 public:
  using OpConversionPattern<tensor::InsertOp>::OpConversionPattern;

  // A helper to create two masks for the case of tensor.insert a cleartext
  // into a ciphertext when the insertion indices are statically known. The
  // scalarMaskValue is the value inserted into the plaintext mask.
  //
  // Returns a pair of scalarMask and destMask.
  std::pair<Value, Value> createMasksFromStaticIndices(
      NewLayoutAttr destLayout, ArrayRef<int64_t> staticIndices,
      tensor::InsertOp op, OpAdaptor adaptor, Value scalarMaskValue,
      ImplicitLocOpBuilder& b) const {
    IntegerRelation destRel = destLayout.getIntegerRelation();
    IntegerRelation fixedRel = fixDomainVars(destRel, staticIndices);
    fixedRel.projectOut(destRel.getVarKindOffset(VarKind::Domain),
                        destRel.getNumDomainVars());

    // Now create a pre-masked packed plainext that has zeros in all but those
    // slots of dest's ciphertext-semantic tensor that the scalar should be
    // "inserted" into. The nonzero slots contain repeated copies of the scalar
    // value.
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getDest().getType());

    arith::ConstantOp zeroTensorOp = arith::ConstantOp::create(
        b, ciphertextSemanticType, b.getZeroAttr(ciphertextSemanticType));
    arith::ConstantOp zeroScalarOp = arith::ConstantOp::create(
        b, ciphertextSemanticType.getElementType(),
        b.getZeroAttr(ciphertextSemanticType.getElementType()));
    arith::ConstantOp oneOp = arith::ConstantOp::create(
        b, ciphertextSemanticType, b.getOneAttr(ciphertextSemanticType));

    // collector.points contains all (ciphertext, slot) pairs that correspond
    // to the locations the scalar should be inserted into.
    PointCollector collector;
    getRangePoints(fixedRel, collector);

    // Iterate over all these points, and for the plaintext mask insert a copy
    // of the scalar, while for the dest mask insert a zero.
    Value scalarMask = zeroTensorOp.getResult();
    Value destMask = oneOp.getResult();
    for (const auto& point : collector.points) {
      SmallVector<Value> indexCsts;
      for (int64_t v : point) {
        auto cst = arith::ConstantIndexOp::create(b, v);
        indexCsts.push_back(cst);
        setMaterializedAttr(cst);
      }

      // Here the scalarMaskValue can be an actual scalar (in the case of a
      // cleartext scalar) or a 1 (in the case of a secret scalar, so that the
      // resulting mask requires an extra mul).
      auto insertIntoScalarMask =
          tensor::InsertOp::create(b, scalarMaskValue, scalarMask, indexCsts);
      auto insertIntoDestMask =
          tensor::InsertOp::create(b, zeroScalarOp, destMask, indexCsts);
      scalarMask = insertIntoScalarMask.getResult();
      destMask = insertIntoDestMask.getResult();
      setMaterializedAttr({insertIntoScalarMask, insertIntoDestMask});
    }

    setMaterializedAttr({zeroTensorOp, zeroScalarOp, oneOp});
    return {scalarMask, destMask};
  }

  // A helper to create two masks for the case of tensor.insert a cleartext
  // into a ciphertext when the insertion indices are not known statically.
  // The scalarMaskValue is the value inserted into the plaintext mask.
  //
  // Returns a pair of scalarMask and destMask.
  std::pair<Value, Value> createMasksFromDynamicIndices(
      NewLayoutAttr destLayout, tensor::InsertOp op, OpAdaptor adaptor,
      Value scalarMaskValue, ImplicitLocOpBuilder& b) const {
    IntegerRelation destRel = destLayout.getIntegerRelation();
    // Here we need to loop over the relation in the same manner as we do
    // codegen for assign_layout, but insert mask entries instead of copying
    // from a data tensor.
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getDest().getType());

    arith::ConstantOp zeroTensorOp = arith::ConstantOp::create(
        b, ciphertextSemanticType, b.getZeroAttr(ciphertextSemanticType));
    arith::ConstantOp zeroScalarOp = arith::ConstantOp::create(
        b, ciphertextSemanticType.getElementType(),
        b.getZeroAttr(ciphertextSemanticType.getElementType()));
    arith::ConstantOp oneOp = arith::ConstantOp::create(
        b, ciphertextSemanticType, b.getOneAttr(ciphertextSemanticType));

    Value scalarMask = zeroTensorOp.getResult();
    Value destMask = oneOp.getResult();
    SmallVector<Value> indices(op.getIndices());

    auto loop = generateLoopWithDynamicIndexCheck(
        b, destRel, {scalarMask, destMask}, indices,
        [&](OpBuilder& builder, Location loc, ValueRange slotIndices,
            ValueRange iterArgs) {
          Value curScalarMask = iterArgs[0];
          Value curDestMask = iterArgs[1];
          auto insertIntoScalarMask = tensor::InsertOp::create(
              b, scalarMaskValue, curScalarMask, slotIndices);
          auto insertIntoDestMask = tensor::InsertOp::create(
              b, zeroScalarOp, curDestMask, slotIndices);
          setMaterializedAttr({insertIntoScalarMask, insertIntoDestMask});
          SmallVector<Value> results = {insertIntoScalarMask.getResult(),
                                        insertIntoDestMask.getResult()};
          return results;
        });
    if (failed(loop)) {
      op.emitError() << "Failed to generate loop nest for layout "
                     << destLayout;
      return {Value(), Value()};
    }

    setMaterializedAttr({zeroTensorOp, zeroScalarOp, oneOp});
    return {loop.value()[0], loop.value()[1]};
  }

  LogicalResult cleartextScalarSecretTensor(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const {
    NewLayoutAttr destLayout = cast<NewLayoutAttr>(
        findAttributeAssociatedWith(adaptor.getDest(), kLayoutAttrName)
            .value());

    Value scalarMask;
    Value destMask;

    SmallVector<Value> indices(op.getIndices());
    auto staticIndicesResult = getConstantIntValues(getAsOpFoldResult(indices));
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getDest().getType());
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (staticIndicesResult.has_value()) {
      SmallVector<int64_t> staticIndices = staticIndicesResult.value();
      std::pair<Value, Value> results = createMasksFromStaticIndices(
          destLayout, staticIndices, op, adaptor, adaptor.getScalar(), b);
      scalarMask = results.first;
      destMask = results.second;
    } else {
      std::pair<Value, Value> results = createMasksFromDynamicIndices(
          destLayout, op, adaptor, adaptor.getScalar(), b);
      scalarMask = results.first;
      destMask = results.second;
    }

    // Mask the dest with the destMask, then add the result to the
    // scalarMask
    StringRef mulOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.muli"
            : "arith.mulf";
    StringRef addOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.addi"
            : "arith.addf";

    Operation* destMul = b.create(OperationState(op.getLoc(), mulOpName,
                                                 {destMask, adaptor.getDest()},
                                                 {ciphertextSemanticType}));
    Operation* finalAdd = b.create(OperationState(
        op.getLoc(), addOpName, {scalarMask, destMul->getResult(0)},
        {ciphertextSemanticType}));
    Value result = finalAdd->getResult(0);
    setMaterializedAttr({destMul, finalAdd});
    setAttributeAssociatedWith(result, kLayoutAttrName, destLayout);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult secretScalarSecretTensor(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const {
    FailureOr<Attribute> scalarLayoutResult =
        findAttributeAssociatedWith(adaptor.getScalar(), kLayoutAttrName);
    FailureOr<Attribute> destLayoutResult =
        findAttributeAssociatedWith(adaptor.getDest(), kLayoutAttrName);

    NewLayoutAttr scalarLayout =
        dyn_cast<NewLayoutAttr>(scalarLayoutResult.value());
    NewLayoutAttr destLayout =
        dyn_cast<NewLayoutAttr>(destLayoutResult.value());

    // Initial support for this kernel requires the sizes of the range to match
    IntegerRelation scalarRel = scalarLayout.getIntegerRelation();
    IntegerRelation destRel = destLayout.getIntegerRelation();
    if (!scalarRel.getRangeSet().isEqual(destRel.getRangeSet())) {
      return op.emitError()
             << "tensor.insert requires scalar and tensor layout to match, but "
                "got scalar layout "
             << scalarLayout << " and tensor layout " << destLayout << "\n";
    }

    Value scalarMask;
    Value destMask;

    SmallVector<Value> indices(op.getIndices());
    auto staticIndicesResult = getConstantIntValues(getAsOpFoldResult(indices));
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getDest().getType());
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // In the cleartextScalar case, we inserted the scalar value directly in
    // the slots of the plaintext mask. Here we insert a 1, and then multiply
    // the resulting mask by the packed scalar ciphertext.
    Value scalarOne = arith::ConstantOp::create(
        b, ciphertextSemanticType.getElementType(),
        b.getOneAttr(ciphertextSemanticType.getElementType()));

    if (staticIndicesResult.has_value()) {
      SmallVector<int64_t> staticIndices = staticIndicesResult.value();
      std::pair<Value, Value> results = createMasksFromStaticIndices(
          destLayout, staticIndices, op, adaptor, scalarOne, b);
      scalarMask = results.first;
      destMask = results.second;
    } else {
      std::pair<Value, Value> results =
          createMasksFromDynamicIndices(destLayout, op, adaptor, scalarOne, b);
      scalarMask = results.first;
      destMask = results.second;
    }

    // Mask the scalarMask with the packed scalar. Mask the dest with the
    // destMask, then add the two results.
    StringRef mulOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.muli"
            : "arith.mulf";
    StringRef addOpName =
        isa<IntegerType>(ciphertextSemanticType.getElementType())
            ? "arith.addi"
            : "arith.addf";

    // Mul the masks to the respective ciphertexts
    Operation* scalarMul = b.create(OperationState(
        op.getLoc(), mulOpName, {scalarMask, adaptor.getScalar()},
        {ciphertextSemanticType}));
    Operation* destMul = b.create(OperationState(op.getLoc(), mulOpName,
                                                 {destMask, adaptor.getDest()},
                                                 {ciphertextSemanticType}));
    Operation* finalAdd = b.create(
        OperationState(op.getLoc(), addOpName,
                       {scalarMul->getResult(0), destMul->getResult(0)},
                       {ciphertextSemanticType}));
    Value result = finalAdd->getResult(0);
    setMaterializedAttr({scalarMul, destMul, finalAdd});
    setAttributeAssociatedWith(result, kLayoutAttrName, destLayout);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    FailureOr<Attribute> scalarLayoutResult =
        findAttributeAssociatedWith(adaptor.getScalar(), kLayoutAttrName);
    FailureOr<Attribute> destLayoutResult =
        findAttributeAssociatedWith(adaptor.getDest(), kLayoutAttrName);

    bool isSecretScalar = succeeded(scalarLayoutResult);
    bool isSecretDest = succeeded(destLayoutResult);

    if (isSecretScalar && isSecretDest) {
      return secretScalarSecretTensor(op, adaptor, rewriter);
    }

    if (!isSecretScalar && isSecretDest) {
      return cleartextScalarSecretTensor(op, adaptor, rewriter);
    }

    if (isSecretScalar && !isSecretDest) {
      return op.emitError() << "dest tensor should have been assigned a layout "
                               "by layout-propagation";
    }

    // cleartext scalar and cleartext tensor means this is a cleartext op
    // that can be elided.
    setMaterializedAttr(op);
    return success();
  }
};

class ConvertCollapseShape
    : public OpConversionPattern<tensor::CollapseShapeOp> {
 public:
  using OpConversionPattern<tensor::CollapseShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::CollapseShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // if the layouts are equal (modulo the rank reduction) then this can be a
    // no-op. check if the layouts are equal after collapsing the dimensions.
    // then if the input and output types are the same, just replace with input.
    SliceVerificationResult res =
        isRankReducedType(op.getSrcType(), op.getResultType());
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(
          op, "Only rank-reduced types are supported for CollapseShapeOp");

    auto srcType = adaptor.getSrc().getType();
    FailureOr<Attribute> tensorLayoutResult =
        findAttributeAssociatedWith(adaptor.getSrc(), kLayoutAttrName);
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    auto tensorLayout = dyn_cast<NewLayoutAttr>(tensorLayoutResult.value());
    if (!tensorLayout) {
      return op.emitError() << "failed to fetch new layout attribute for input";
    }
    FailureOr<Attribute> resultLayoutResult =
        findAttributeAssociatedWith(op.getResult(), kLayoutAttrName);
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    NewLayoutAttr resultLayout =
        dyn_cast<NewLayoutAttr>(resultLayoutResult.value());
    if (!resultLayout) {
      return op.emitError() << "failed to fetch new layout attribute for input";
    }
    Type resultType = getTypeConverter()->convertType(op.getResult());

    if (resultType != srcType) {
      return rewriter.notifyMatchFailure(
          op, "result type does not match input type");
    }

    auto srcRelation = tensorLayout.getIntegerRelation();
    auto collapsedRelation = collapseDimensions(srcRelation, op.getSrcType(),
                                                op.getReassociationIndices());
    if (!collapsedRelation.isEqual(resultLayout.getIntegerRelation())) {
      return rewriter.notifyMatchFailure(
          op, "result layout is not equal to input layout");
    }

    // Put in a no-op unrealized conversion cast operation to persist the new
    // attribute for downstream ops.
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), resultType, adaptor.getSrc());
    setMaterializedAttr(castOp);
    setAttributeAssociatedWith(castOp.getResult(0), kLayoutAttrName,
                               resultLayout);

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

class ConvertExpandShape : public OpConversionPattern<tensor::ExpandShapeOp> {
 public:
  using OpConversionPattern<tensor::ExpandShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::ExpandShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // if the layouts are equal (modulo the rank reduction) then this can be a
    // no-op. check if the layouts are equal after collapsing the dimensions.
    // then if the input and output types are the same, just replace with input.
    SliceVerificationResult res =
        isRankReducedType(op.getResultType(), op.getSrcType());
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(
          op, "Only rank-reduced types are supported for CollapseShapeOp");

    FailureOr<Attribute> resultLayoutResult =
        findAttributeAssociatedWith(op.getResult(), kLayoutAttrName);
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    NewLayoutAttr resultLayout =
        dyn_cast<NewLayoutAttr>(resultLayoutResult.value());
    if (!resultLayout) {
      return op.emitError() << "failed to fetch new layout attribute for input";
    }
    Type resultType = getTypeConverter()->convertType(op.getResult());

    auto srcType = adaptor.getSrc().getType();
    FailureOr<Attribute> sourceLayoutResult =
        findAttributeAssociatedWith(adaptor.getSrc(), kLayoutAttrName);
    if (failed(sourceLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    auto sourceLayout = dyn_cast<NewLayoutAttr>(sourceLayoutResult.value());
    if (!sourceLayout) {
      return op.emitError() << "failed to fetch new layout attribute for input";
    }

    if (resultType != srcType) {
      return rewriter.notifyMatchFailure(
          op, "result type does not match input type");
    }

    auto srcRelation = sourceLayout.getIntegerRelation();
    auto expandedRelation = expandDimensions(srcRelation, op.getResultType(),
                                             op.getReassociationIndices());
    if (!expandedRelation.isEqual(resultLayout.getIntegerRelation())) {
      return rewriter.notifyMatchFailure(
          op, "result layout is not equal to input layout");
    }

    // Put in a no-op unrealized conversion cast operation to persist the new
    // attribute for downstream ops.
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), resultType, adaptor.getSrc());
    setMaterializedAttr(castOp);
    setAttributeAssociatedWith(castOp.getResult(0), kLayoutAttrName,
                               resultLayout);

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct DropRotateUnitDims : OpRewritePattern<tensor_ext::RotateOp> {
  using OpRewritePattern<tensor_ext::RotateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor_ext::RotateOp rotateOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> operandUnitDims =
        getUnitDims(rotateOp.getTensor().getType());
    if (operandUnitDims.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "no unit dims to drop\n");
      return failure();
    }

    SmallVector<Value> collapsedOperands =
        collapseOperands(rewriter, {rotateOp.getTensor()}, operandUnitDims);

    tensor_ext::RotateOp collapsedOp = tensor_ext::RotateOp::create(
        rewriter, rotateOp.getLoc(), collapsedOperands[0], rotateOp.getShift());
    rewriter.replaceOp(rotateOp, expandResult(rewriter, collapsedOp.getResult(),
                                              rotateOp.getOutput().getType(),
                                              operandUnitDims));
    return success();
  }
};

struct DropRotateAndReduceUnitDims
    : OpRewritePattern<tensor_ext::RotateAndReduceOp> {
  using OpRewritePattern<tensor_ext::RotateAndReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor_ext::RotateAndReduceOp rotateOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> operandUnitDims =
        getUnitDims(rotateOp.getTensor().getType());
    if (operandUnitDims.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "no unit dims to drop\n");
      return failure();
    }

    SmallVector<Value> collapsedOperands =
        collapseOperands(rewriter, {rotateOp.getTensor()}, operandUnitDims);

    auto collapsedOp = tensor_ext::RotateAndReduceOp::create(
        rewriter, rotateOp.getLoc(), collapsedOperands[0],
        rotateOp.getPlaintexts(), rotateOp.getPeriod().getZExtValue(),
        rotateOp.getSteps().getZExtValue(), rotateOp.getReduceOp());
    rewriter.replaceOp(rotateOp, expandResult(rewriter, collapsedOp.getResult(),
                                              rotateOp.getOutput().getType(),
                                              operandUnitDims));
    return success();
  }
};

struct DropElementwiseUnitDims : OpTraitRewritePattern<OpTrait::Elementwise> {
  explicit DropElementwiseUnitDims(MLIRContext* context)
      : OpTraitRewritePattern(context) {}

  LogicalResult matchAndRewrite(mlir::Operation* op,
                                PatternRewriter& rewriter) const override {
    // Ensure that all operands and results have the same type.
    SmallVector<Type> operandAndResultTypes =
        llvm::to_vector(op->getOperandTypes());
    operandAndResultTypes.append(op->getResultTypes().begin(),
                                 op->getResultTypes().end());
    if (!llvm::all_equal(operandAndResultTypes) || op->getNumOperands() == 0 ||
        op->getNumResults() != 1) {
      return failure();
    }

    auto tensorType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!tensorType) {
      return failure();
    }

    SmallVector<int64_t> operandUnitDims = getUnitDims(tensorType);
    if (operandUnitDims.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "no unit dims to drop\n");
      return failure();
    }

    SmallVector<Value> collapsedOperands = collapseOperands(
        rewriter, llvm::to_vector(op->getOperands()), operandUnitDims);

    Type resultType = collapsedOperands[0].getType();
    Operation* collapsedOp = rewriter.create(OperationState(
        op->getLoc(), op->getName().getStringRef(), collapsedOperands,
        resultType, op->getAttrs(), op->getSuccessors()));

    rewriter.replaceOp(
        op, expandResult(rewriter, collapsedOp->getResults()[0],
                         cast<RankedTensorType>(op->getResult(0).getType()),
                         operandUnitDims));
    return success();
  }
};

struct ConvertToCiphertextSemantics
    : impl::ConvertToCiphertextSemanticsBase<ConvertToCiphertextSemantics> {
  using ConvertToCiphertextSemanticsBase::ConvertToCiphertextSemanticsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    int64_t ctSize = ciphertextSize;
    LayoutMaterializationTypeConverter typeConverter =
        LayoutMaterializationTypeConverter(ctSize);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      return isa<ModuleOp>(op) || hasMaterializedAttr(op);
    });

    patterns.add<ConvertGeneric,
                 // tensor_ext ops
                 ConvertConvertLayout, ConvertConvertLayoutNewLayout,
                 // linalg ops
                 ConvertLinalgReduce, ConvertLinalgMatvec,
                 ConvertLinalgMatvecNewLayout, ConvertLinalgConv2D,
                 // tensor ops
                 ConvertTensorExtract, ConvertTensorInsert,
                 ConvertTensorExtractNewLayout, ConvertTensorInsertNewLayout,
                 ConvertCollapseShape, ConvertExpandShape,
                 // default
                 ConvertAnyAddingMaterializedAttr>(typeConverter, context);
    patterns.add<ConvertFunc, ConvertAssignLayout>(typeConverter, context,
                                                   ciphertextSize);

    ConversionConfig config;
    config.buildMaterializations = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      return signalPassFailure();
    }

    // Decompose tensor.concat into repeated tensor.insert_slice ops.
    // Note ConvertAssignLayout generates tensor.concat
    RewritePatternSet cleanupPatterns2(context);
    tensor::populateDecomposeTensorConcatPatterns(cleanupPatterns2);
    // Drop unit dimensions for tensor_ext ops that require 1-D tensors (i.e.
    // rotation ops) and elementwise ops.
    cleanupPatterns2.add<DropRotateUnitDims, DropRotateAndReduceUnitDims,
                         DropElementwiseUnitDims>(context);
    mlir::tensor::populateReassociativeReshapeFoldingPatterns(cleanupPatterns2);
    // Folding here will remove any unrealized conversion cast ops that were
    // inserted to persist new layouts.
    if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns2)))) {
      return signalPassFailure();
    }

    clearAttrs(module, kLayoutAttrName);
    clearAttrs(module, kMaterializedAttrName);
  }
};

}  // namespace heir
}  // namespace mlir
