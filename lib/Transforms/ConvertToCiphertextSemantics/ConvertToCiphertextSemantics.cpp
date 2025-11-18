#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "lib/Transforms/DropUnitDims/DropUnitDims.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
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
using ::mlir::heir::kernel::ArithmeticDagNode;
using ::mlir::heir::kernel::implementHaleviShoup;
using ::mlir::heir::kernel::IRMaterializingVisitor;
using ::mlir::heir::kernel::SSAValue;
using presburger::IntegerRelation;
using presburger::VarKind;
using tensor_ext::LayoutAttr;

auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
auto& kMaterializedAttrName = "tensor_ext.layout_materialized";
auto& kOriginalTypeAttrName =
    tensor_ext::TensorExtDialect::kOriginalTypeAttrName;

NamedAttribute getArithFMF(OpBuilder& builder) {
  return mlir::NamedAttribute(
      mlir::StringAttr::get(builder.getContext(), "fastmath"),
      mlir::arith::FastMathFlagsAttr::get(
          builder.getContext(),
          mlir::arith::FastMathFlags::nsz | mlir::arith::FastMathFlags::nnan));
}

IntegerRelation restrictRelationToSlice(const IntegerRelation& relation,
                                        llvm::ArrayRef<int64_t> offsets,
                                        llvm::ArrayRef<int64_t> sizes) {
  auto result = relation.clone();
  // Set the domain vars to the range specified by the offset and size.
  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    // d0 - offset >= 0
    SmallVector<int64_t> leConstraint(result->getNumCols(), 0);
    leConstraint[dim + result->getVarKindOffset(VarKind::Domain)] = 1;
    leConstraint.back() = -offset;
    result->addInequality(leConstraint);
    // offset + size - d0 - 1 >= 0
    SmallVector<int64_t> geConstraint(result->getNumCols(), 0);
    geConstraint[dim + result->getVarKindOffset(VarKind::Domain)] = -1;
    geConstraint.back() = offset + sizes[dim] - 1;
    result->addInequality(geConstraint);
  }
  return *result;
}

// tensor_ext.remap constrains its input and output types to be the same,
// i.e., remap occurs within one set of ciphertexts. The output of an
// extract_slice, however, may have a layout that has fewer ciphertexts
// in it. For example, extracting one row from a data-semantic matrix that
// is packed with one row per ciphertext would result in a single output
// ciphertext, and the expected layout of the result will reflect that.
// To bridge this gap, kernels must post-processes the remap's output to
// extract the subset ciphertexts relevant to the layout of the output
// slice.
Operation* remapAndExtractResult(ImplicitLocOpBuilder& builder, Value input,
                                 LayoutAttr resultLayout,
                                 RankedTensorType resultType) {
  auto remapOp = tensor_ext::RemapOp::create(builder, input, resultLayout);

  SmallVector<OpFoldResult> strides(2, builder.getIndexAttr(1));
  SmallVector<OpFoldResult> offsets(2, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  sizes.push_back(builder.getIndexAttr(resultType.getDimSize(0)));
  sizes.push_back(builder.getIndexAttr(resultType.getDimSize(1)));
  auto extractRemap = tensor::ExtractSliceOp::create(
      builder, resultType, remapOp.getResult(), offsets, sizes, strides);
  return extractRemap;
}

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
      auto innerType = type.getValueType();

      FailureOr<Type> convertedInnerType = materializeLayout(
          getElementTypeOrSelf(innerType), attr, getCiphertextSize());
      if (failed(convertedInnerType)) return std::nullopt;
      return secret::SecretType::get(convertedInnerType.value());
    });
    addConversion(
        [this](RankedTensorType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeLayout(getElementTypeOrSelf(type), attr,
                                   getCiphertextSize());
        });
    addConversion(
        [this](IntegerType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeLayout(type, attr, getCiphertextSize());
        });
    addConversion(
        [this](FloatType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeLayout(type, attr, getCiphertextSize());
        });
    addConversion(
        [this](IndexType type, LayoutAttr attr) -> std::optional<Type> {
          return materializeLayout(type, attr, getCiphertextSize());
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
  ConvertFunc(const ContextAwareTypeConverter& converter, MLIRContext* context)
      : ContextAwareFuncConversion(converter, context) {}

  LogicalResult finalizeFuncOpModification(
      func::FuncOp op, ArrayRef<Type> oldArgTypes,
      ArrayRef<Type> oldResultTypes, PatternRewriter& rewriter) const override {
    // Replace layout arg attrs with secret.original_type arg attrs This is
    // necessary so that later encoding/decoding functions can know what the
    // original type of the tensor was and how it was encoded.
    rewriter.modifyOpInPlace(op, [&] {
      setMaterializedAttr(op);
      for (int i = 0; i < op.getNumArguments(); ++i) {
        auto layoutAttr = op.getArgAttr(i, kLayoutAttrName);
        if (!layoutAttr || !isa<LayoutAttr>(layoutAttr)) {
          continue;
        }

        op.setArgAttr(i, kOriginalTypeAttrName,
                      tensor_ext::OriginalTypeAttr::get(
                          getContext(), maybeExtractSecretType(oldArgTypes[i]),
                          layoutAttr));
      }

      for (int i = 0; i < op.getNumResults(); ++i) {
        auto layoutAttr = op.getResultAttr(i, kLayoutAttrName);
        if (!layoutAttr || !isa<LayoutAttr>(layoutAttr)) {
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

struct ConvertSecretGeneric : public ConvertAnyContextAware<secret::GenericOp> {
 public:
  ConvertSecretGeneric(const ContextAwareTypeConverter& converter,
                       MLIRContext* context)
      : ConvertAnyContextAware(converter, context) {
    setDebugName("ConvertGeneric");
  }

  LogicalResult finalizeOpModification(
      secret::GenericOp op,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] { setMaterializedAttr(op); });
    return success();
  };
};

// Convert an op generically, marking it as materialized. Lowest priority
// because it is only meant to handle ops that don't have special
// materialization rules.
struct ConvertAnyAddingMaterializedAttr : public ConvertAnyContextAware<> {
  ConvertAnyAddingMaterializedAttr(const ContextAwareTypeConverter& converter,
                                   MLIRContext* context)
      : ConvertAnyContextAware(converter, context, /*benefit=*/0) {
    setDebugName("ConvertAnyAddingMaterializedAttr");
  }

  LogicalResult finalizeOpModification(
      Operation* op,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] { setMaterializedAttr(op); });
    return success();
  };
};

class ConvertAssignLayout
    : public ContextAwareOpConversionPattern<tensor_ext::AssignLayoutOp> {
 public:
  ConvertAssignLayout(const ContextAwareTypeConverter& typeConverter,
                      mlir::MLIRContext* context, int64_t ciphertextSize)
      : ContextAwareOpConversionPattern<tensor_ext::AssignLayoutOp>(
            typeConverter, context),
        ciphertextSize(ciphertextSize) {}

  LogicalResult matchAndRewrite(
      tensor_ext::AssignLayoutOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
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
    : public ContextAwareOpConversionPattern<tensor_ext::ConvertLayoutOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor_ext::ConvertLayoutOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor_ext::ConvertLayoutOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
    LayoutAttr fromLayout = dyn_cast<LayoutAttr>(op.getFromLayout());
    LayoutAttr toLayout = dyn_cast<LayoutAttr>(op.getToLayout());
    if (!fromLayout || !toLayout) {
      return failure();
    }

    // This is persisted as an operation rather than lowered eagerly to a shift
    // network so as to allow VosVosErkinShiftNetworks to cache its work across
    // multiple instances of the same conversion.
    std::shared_ptr<IntegerRelation> composedLayout =
        fromLayout.getIntegerRelation().clone();
    composedLayout->inverse();
    composedLayout->compose(toLayout.getIntegerRelation());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    LayoutAttr newLayoutAttr =
        LayoutAttr::getFromIntegerRelation(getContext(), *composedLayout);
    auto resultCiphertextSemanticType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType(), toLayout));
    auto remapAndExtract = remapAndExtractResult(
        b, adaptor.getValue(), newLayoutAttr, resultCiphertextSemanticType);

    setMaterializedAttr(remapAndExtract);
    setAttributeAssociatedWith(remapAndExtract->getResults()[0],
                               kLayoutAttrName, toLayout);
    rewriter.replaceOp(op, remapAndExtract->getResults()[0]);
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

class ConvertLinalgReduce
    : public ContextAwareOpConversionPattern<linalg::ReduceOp> {
 public:
  using ContextAwareOpConversionPattern<
      linalg::ReduceOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::ReduceOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
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

    // TODO(#2254): Implement a proper kernel
    // rewriter.replaceOp(op, result);
    // return success();
    return failure();
  }
};

struct ConvertLinalgMatvecLayout
    : public ContextAwareOpConversionPattern<linalg::MatvecOp> {
 public:
  using ContextAwareOpConversionPattern<
      linalg::MatvecOp>::ContextAwareOpConversionPattern;

  ConvertLinalgMatvecLayout(
      const ContextAwareTypeConverter& contextAwareTypeConverter,
      MLIRContext* context)
      : ContextAwareOpConversionPattern(contextAwareTypeConverter, context,
                                        /*benefit=*/10) {}

  LayoutAttr getLayoutAttr(Value value) const {
    auto layoutLookup = getTypeConverter()->getContextualAttr(value);
    if (failed(layoutLookup)) {
      return nullptr;
    }
    return dyn_cast<LayoutAttr>(layoutLookup.value());
  }

  bool supportsHaleviShoup(linalg::MatvecOp op, OpAdaptor adaptor) const {
    Value matrix = adaptor.getInputs()[0];
    auto matrixType = cast<RankedTensorType>(matrix.getType());

    // If one of these dimensions is not a power of two, then we can't do
    // the Halevi-Shoup or Squat Packing Matrix Multiplication conversion.
    auto dimensions = matrixType.getShape();
    int64_t numRows = dimensions[0];
    int64_t numCols = dimensions[1];
    bool isPowerOfTwoDims = isPowerOfTwo(numRows) && isPowerOfTwo(numCols);

    // TODO(#1578): If the matrix has more rows than columns, what kernel
    // should be used?
    bool dimensionsCompatible = numRows <= numCols;

    auto kernelAttr = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    bool isMatvecDiagonal =
        kernelAttr && kernelAttr.getName() == KernelName::MatvecDiagonal;

    LLVM_DEBUG(llvm::dbgs()
               << "supports matvec with halevi-shoup: isPowerOfTwoDims="
               << isPowerOfTwoDims
               << " isDimensionsCompatible=" << dimensionsCompatible
               << " isMatvecDiagonal=" << isMatvecDiagonal << "\n");

    return isPowerOfTwoDims && dimensionsCompatible && isMatvecDiagonal;
  }

  void haleviShoupKernel(
      linalg::MatvecOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const {
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
    IRMaterializingVisitor visitor(
        b, input.getType(),
        [&](Operation* createdOp) { setMaterializedAttr(createdOp); });
    Value finalOutput = implementedKernel->visit(visitor);

    auto layoutAttr = cast<LayoutAttr>(op->getAttr(kLayoutAttrName));
    auto* finalOutputOp = finalOutput.getDefiningOp();
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
      ContextAwareConversionPatternRewriter& rewriter) const final {
    Value matrix = adaptor.getInputs()[0];
    Value vector = adaptor.getInputs()[1];
    LayoutAttr vectorLayout = getLayoutAttr(vector);
    LayoutAttr matrixLayout = getLayoutAttr(matrix);

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

struct ConvertLinalgConv2D
    : public ContextAwareOpConversionPattern<linalg::Conv2DOp> {
 public:
  using ContextAwareOpConversionPattern<
      linalg::Conv2DOp>::ContextAwareOpConversionPattern;

  ConvertLinalgConv2D(
      const ContextAwareTypeConverter& contextAwareTypeConverter,
      MLIRContext* context)
      : ContextAwareOpConversionPattern(contextAwareTypeConverter, context,
                                        /*benefit=*/10) {}

  LayoutAttr getLayoutAttr(Value value) const {
    auto layoutLookup = getTypeConverter()->getContextualAttr(value);
    if (failed(layoutLookup)) {
      return nullptr;
    }
    return dyn_cast<LayoutAttr>(layoutLookup.value());
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

  void haleviShoupKernel(
      linalg::Conv2DOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "Converting linalg.conv2d op with halevi shoup kernel: " << op
               << "\n");

    TypedValue<RankedTensorType> data =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[0]);
    SSAValue vectorLeaf(data);
    TypedValue<RankedTensorType> matrix =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[1]);
    SSAValue matrixLeaf(matrix);

    // The original matrix shape is the shape of the expanded filter before
    // diagonalization. This is 28x28 for LeNet
    RankedTensorType expandedMatrixType = get2dConvFilterExpandedType(
        cast<RankedTensorType>(op.getInputs()[1].getType()),
        cast<RankedTensorType>(op.getInputs()[0].getType()), /*padding=*/0);

    // Get non-zero diagonals of the diagonalized expanded filter matrix.
    LayoutAttr filterLayout = getLayoutAttr(adaptor.getInputs()[1]);
    auto filterRelation = filterLayout.getIntegerRelation();
    PointCollector collector;
    getRangePoints(filterRelation, collector);
    std::map<int, bool> nonZeroDiagonals;
    for (auto point : collector.points) {
      nonZeroDiagonals[point[0]] = true;
    }
    for (auto [ct, val] : nonZeroDiagonals) {
      std::cout << ct << ", ";
    }
    std::cout << "\n";
    std::cout << "nonZero diagonal size: " << nonZeroDiagonals.size() << "\n";

    std::shared_ptr<ArithmeticDagNode<SSAValue>> implementedKernel =
        implementHaleviShoup(vectorLeaf, matrixLeaf,
                             expandedMatrixType.getShape(), nonZeroDiagonals);

    rewriter.setInsertionPointAfter(op);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    IRMaterializingVisitor visitor(
        b, data.getType(),
        [&](Operation* createdOp) { setMaterializedAttr(createdOp); });
    Value finalOutput = implementedKernel->visit(visitor);

    auto layoutAttr = cast<LayoutAttr>(op->getAttr(kLayoutAttrName));
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
      ContextAwareConversionPatternRewriter& rewriter) const final {
    Value data = adaptor.getInputs().front();
    Value filter = adaptor.getInputs().back();
    LayoutAttr dataLayout = getLayoutAttr(data);
    LayoutAttr filterLayout = getLayoutAttr(filter);

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

Value makeMask(ContextAwareConversionPatternRewriter& rewriter, Location loc,
               Value index, RankedTensorType ciphertextSemanticType) {
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

Value makeInverseMask(ContextAwareConversionPatternRewriter& rewriter,
                      Location loc, Value index,
                      RankedTensorType ciphertextSemanticType) {
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
          indices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), exprs[i]));
        }

        // If statement to check that the current set of exprs matches
        // the dynamic insertion indices.
        Value exprsMatchIndices;

        // conjunction over all exprs[i] == indices[i] where i is a domain
        // variable.
        int domainStart = relation.getVarKindOffset(VarKind::Domain);
        for (int i = domainStart; i < domainStart + relation.getNumDomainVars();
             ++i) {
          auto eqOp = arith::CmpIOp::create(
              b, arith::CmpIPredicate::eq,
              arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                         exprs[i]),
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

class ConvertTensorExtractLayout
    : public ContextAwareOpConversionPattern<tensor::ExtractOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::ExtractOp>::ContextAwareOpConversionPattern;

  struct MaskResult {
    Value mask;
    IntegerRelation maskLayout;
  };

  // A helper to create a mask for the case of tensor.extract from a ciphertext
  // when the extraction indices are statically known.
  MaskResult createMaskFromStaticIndices(LayoutAttr tensorLayout,
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
      ContextAwareConversionPatternRewriter& rewriter) const final {
    // Mask the tensor at the given index, rotate it to the first position, and
    // then ensure its layout matches the expected output layout with a
    // layout_conversion op.
    FailureOr<Attribute> tensorLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getTensor());
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    FailureOr<Attribute> resultLayoutResult =
        getTypeConverter()->getContextualAttr(op.getResult());
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }

    LayoutAttr tensorLayout = cast<LayoutAttr>(tensorLayoutResult.value());
    LayoutAttr resultLayout = cast<LayoutAttr>(resultLayoutResult.value());
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
      //
      // TODO(#2257): Support dynamic indices in tensor.extract
      return op.emitError() << "tensor.extract with dynamic cleartext indices "
                               "not supported yet";
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
        LayoutAttr::getFromIntegerRelation(op.getContext(),
                                           maskResult.maskLayout),
        resultLayout);

    // Intentionally don't set materialized attr on convert_layout so it will be
    // processed by a subsequent pattern.
    setMaterializedAttr(mulOp);
    // The layout conversion may be folded away, so the mul op also needs an
    // attribute
    setAttributeAssociatedWith(mulOp->getResults()[0], kLayoutAttrName,
                               resultLayout);
    setAttributeAssociatedWith(convertLayoutOp.getResult(), kLayoutAttrName,
                               resultLayout);
    rewriter.replaceOp(op, convertLayoutOp);
    return success();
  }
};

class ConvertTensorInsertSlice
    : public ContextAwareOpConversionPattern<tensor::InsertSliceOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::InsertSliceOp>::ContextAwareOpConversionPattern;

  // A helper to create two masks for the case of tensor.insert_slice a
  // cleartext into a ciphertext when the insertion indices are statically
  // known. The scalarMaskValue is the value inserted into the plaintext mask.
  //
  // Returns a pair of scalarMask and destMask.
  std::pair<std::vector<Value>, std::vector<Value>>
  createMasksFromStaticIndicesAtCt(const PointCollector& collector,
                                   RankedTensorType destCiphertextSemanticType,
                                   int64_t scalarMaskValue,
                                   ImplicitLocOpBuilder& b) const {
    // Now create pre-masked packed plainexts that have zeros in all but those
    // slots of dest's ciphertext-semantic tensor that the scalar should be
    // "inserted" into. The nonzero slots contain repeated copies of the scalar
    // value.
    auto numCiphertexts = destCiphertextSemanticType.getDimSize(0);
    Type elementType = destCiphertextSemanticType.getElementType();
    RankedTensorType singleCiphertextType = RankedTensorType::get(
        {1, destCiphertextSemanticType.getDimSize(1)}, elementType);

    // Create dense constants for the scalar mask and dest mask.
    std::vector<SmallVector<Attribute>> scalarMaskAttrs(
        numCiphertexts,
        SmallVector<Attribute>(singleCiphertextType.getNumElements(),
                               b.getZeroAttr(elementType)));
    std::vector<SmallVector<Attribute>> destMaskAttrs(
        numCiphertexts,
        SmallVector<Attribute>(singleCiphertextType.getNumElements(),
                               b.getOneAttr(elementType)));
    Attribute scalarMaskAttr =
        isa<IntegerType>(elementType)
            ? cast<Attribute>(b.getIntegerAttr(elementType, scalarMaskValue))
            : b.getFloatAttr(elementType, scalarMaskValue);

    // Iterate over all these points for each ciphertext, and for the plaintext
    // mask insert a copy of the scalar, while for the dest mask insert a zero.
    for (auto& point : collector.points) {
      auto ctIndex = point[0];
      auto slotIndex = point[1];

      // Here the scalarMaskValue can be an actual scalar (in the case of a
      // cleartext scalar) or a 1 (in the case of a secret scalar, so that the
      // resulting mask requires an extra mul).
      scalarMaskAttrs[ctIndex][slotIndex] = scalarMaskAttr;
      destMaskAttrs[ctIndex][slotIndex] = b.getZeroAttr(elementType);
    }

    std::vector<Value> scalarMasks;
    std::vector<Value> destMasks;
    for (auto ct = 0; ct < numCiphertexts; ++ct) {
      arith::ConstantOp scalarMask = arith::ConstantOp::create(
          b, singleCiphertextType,
          DenseElementsAttr::get(singleCiphertextType, scalarMaskAttrs[ct]));
      arith::ConstantOp destMask = arith::ConstantOp::create(
          b, singleCiphertextType,
          DenseElementsAttr::get(singleCiphertextType, destMaskAttrs[ct]));

      setMaterializedAttr({scalarMask, destMask});
      scalarMasks.push_back(scalarMask.getResult());
      destMasks.push_back(destMask.getResult());
    }

    return {scalarMasks, destMasks};
  }

  LogicalResult secretScalarSecretTensor(
      tensor::InsertSliceOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const {
    MLIRContext* ctx = op.getContext();

    FailureOr<Attribute> scalarLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getSource());
    FailureOr<Attribute> destLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getDest());

    LayoutAttr scalarLayout = cast<LayoutAttr>(scalarLayoutResult.value());
    LayoutAttr destLayout = cast<LayoutAttr>(destLayoutResult.value());

    IntegerRelation scalarRel = scalarLayout.getIntegerRelation();
    IntegerRelation destRel = destLayout.getIntegerRelation();

    // Initial support for this kernel requires that the scalar source slice
    // changes layout to match the destination tensor layout at the subspace
    // defined by the slicing.
    if (!scalarRel.getRangeSet().isSubsetOf(destRel.getRangeSet())) {
      return op.emitError()
             << "tensor.insert requires slice layout to be a subset of the "
                "dest layout, but "
                "got slice layout "
             << scalarLayout << " and tensor layout " << destLayout << "\n";
    }

    SmallVector<int64_t> staticStrides =
        SmallVector<int64_t>(op.getStaticStrides());
    if (!llvm::all_of(staticStrides, [](int64_t s) { return s == 1; })) {
      return op.emitError()
             << "only unit strides of a tensor.insert_slice are supported";
    }

    // Convert the scalar layout to the destination layout subspace.
    auto maybeSliceLayout = getSliceInsertionRelation(
        op.getSourceType(), op.getResultType(),
        SmallVector<int64_t>(op.getStaticOffsets()),
        SmallVector<int64_t>(op.getStaticSizes()), staticStrides);
    if (failed(maybeSliceLayout)) {
      return op.emitError() << "failed to get layout for insert slice";
    }
    auto sliceInsertionLayout = maybeSliceLayout.value();
    sliceInsertionLayout.compose(destRel);

    // If we insert at an offset other than the zero, we may need to reindex
    // the resulting ciphertext so that it is indexed at 0. Shift the range of
    // the ciphertext output var back to index at 0.
    auto ctLowerBound = sliceInsertionLayout.getConstantBound64(
        presburger::BoundType::LB,
        sliceInsertionLayout.getVarKindOffset(presburger::VarKind::Range));
    if (!ctLowerBound) {
      return op.emitError()
             << "failed to get constant bound on ciphertext index";
    }
    // Shift range ct var by -ctLowerBound.
    auto ctVar =
        sliceInsertionLayout.getVarKindOffset(presburger::VarKind::Range);
    auto shiftedSliceInsertionLayout =
        shiftVar(sliceInsertionLayout, ctVar, -ctLowerBound.value());

    // In case the slice is already in a layout compatible with the destination
    // layout, we don't need to insert a conversion.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value convertedSource = adaptor.getSource();
    if (!scalarRel.isEqual(shiftedSliceInsertionLayout)) {
      LayoutAttr newScalarLayout =
          LayoutAttr::getFromIntegerRelation(ctx, shiftedSliceInsertionLayout);
      LLVM_DEBUG(llvm::dbgs()
                 << "converting insert_slice source from " << scalarLayout
                 << " to " << newScalarLayout << "\n");
      convertedSource = tensor_ext::ConvertLayoutOp::create(
          b, op.getLoc(), adaptor.getSource(), scalarLayout, newScalarLayout);
    }

    // For each ciphertext in the destination, mask the points originating from
    // the slice with ones and the remaining points from the destination with
    // ones. To do this, we collect all points of the slice in the destination
    // layout.
    IntegerRelation sliceOfDestRelation = restrictRelationToSlice(
        destRel, op.getStaticOffsets(), op.getStaticSizes());
    // collector.points contains all (ciphertext, slot) pairs that correspond
    // to the locations the scalar should be inserted into.
    RankedTensorType ciphertextSemanticType =
        cast<RankedTensorType>(adaptor.getDest().getType());
    PointCollector collector;
    getRangePoints(sliceOfDestRelation, collector);
    std::pair<std::vector<Value>, std::vector<Value>> results =
        createMasksFromStaticIndicesAtCt(collector, ciphertextSemanticType, 1,
                                         b);

    Value result =
        tensor::EmptyOp::create(b, ciphertextSemanticType.getShape(),
                                ciphertextSemanticType.getElementType());
    auto numCt = ciphertextSemanticType.getDimSize(0);
    auto slots = ciphertextSemanticType.getDimSize(1);
    for (auto ct = 0; ct < numCt; ++ct) {
      Value scalarMask = results.first[ct];
      Value destMask = results.second[ct];

      // Mask the scalarMask with the packed scalar. Mask the dest with the
      // destMask, then add the two results.
      SmallVector<OpFoldResult> ctOffsets;
      ctOffsets.push_back(rewriter.getIndexAttr(ct));
      ctOffsets.push_back(rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      sizes.push_back(b.getIndexAttr(1));
      sizes.push_back(b.getIndexAttr(slots));
      SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
      Operation* extractedDest = tensor::ExtractSliceOp::create(
          b, op.getLoc(), cast<RankedTensorType>(convertedSource.getType()),
          adaptor.getDest(), ctOffsets, sizes, strides);
      Operation* scalarMul = makeAppropriatelyTypedMulOp(
          b, op.getLoc(), scalarMask, convertedSource, {getArithFMF(b)});
      Operation* destMul = makeAppropriatelyTypedMulOp(
          b, op.getLoc(), destMask, extractedDest->getResult(0),
          {getArithFMF(b)});
      Operation* finalAdd =
          makeAppropriatelyTypedAddOp(b, op.getLoc(), scalarMul->getResult(0),
                                      destMul->getResult(0), {getArithFMF(b)});

      // Insert the final result into the ciphertext at position ct.
      Operation* insertOp = tensor::InsertSliceOp::create(
          b, finalAdd->getResult(0), result, ctOffsets, sizes, strides);
      setMaterializedAttr(
          {extractedDest, scalarMul, destMul, finalAdd, insertOp});
      result = insertOp->getResult(0);
    }

    setAttributeAssociatedWith(result, kLayoutAttrName, destLayout);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult matchAndRewrite(
      tensor::InsertSliceOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
    FailureOr<Attribute> sourceLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getSource());
    FailureOr<Attribute> destLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getDest());

    bool isSecretScalar = succeeded(sourceLayoutResult);
    bool isSecretDest = succeeded(destLayoutResult);

    if (isSecretScalar && isSecretDest) {
      return secretScalarSecretTensor(op, adaptor, rewriter);
    }

    if (!isSecretScalar && isSecretDest) {
      return op.emitError()
             << "unsupported case of inserting cleartext into secret tensor";
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

class ConvertTensorInsertLayout
    : public ContextAwareOpConversionPattern<tensor::InsertOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::InsertOp>::ContextAwareOpConversionPattern;

  // A helper to create two masks for the case of tensor.insert a cleartext
  // into a ciphertext when the insertion indices are statically known. The
  // scalarMaskValue is the value inserted into the plaintext mask.
  //
  // Returns a pair of scalarMask and destMask.
  std::pair<Value, Value> createMasksFromStaticIndices(
      LayoutAttr destLayout, ArrayRef<int64_t> staticIndices,
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
      LayoutAttr destLayout, tensor::InsertOp op, OpAdaptor adaptor,
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
          SmallVector<Value> slotIndicesAsIndex;
          for (Value index : slotIndices) {
            slotIndicesAsIndex.push_back(arith::IndexCastOp::create(
                builder, loc, builder.getIndexType(), index));
          }
          auto insertIntoScalarMask = tensor::InsertOp::create(
              b, scalarMaskValue, curScalarMask, slotIndicesAsIndex);
          auto insertIntoDestMask = tensor::InsertOp::create(
              b, zeroScalarOp, curDestMask, slotIndicesAsIndex);
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
      ContextAwareConversionPatternRewriter& rewriter) const {
    LayoutAttr destLayout = cast<LayoutAttr>(
        getTypeConverter()->getContextualAttr(adaptor.getDest()).value());

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
      ContextAwareConversionPatternRewriter& rewriter) const {
    FailureOr<Attribute> scalarLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getScalar());
    FailureOr<Attribute> destLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getDest());

    LayoutAttr scalarLayout = dyn_cast<LayoutAttr>(scalarLayoutResult.value());
    LayoutAttr destLayout = dyn_cast<LayoutAttr>(destLayoutResult.value());

    // Initial support for this kernel requires the sizes of the range to
    // match. Ideally we could be smarter here, for example to take a scalar
    // layout which is a single slot in a single ciphertext, and insert it into
    // the correct slot of the single ciphertext of the dest tensor via a mask
    // and rotate. But this kernel is not expected to be used, so we can instead
    // incur the cost of a layout conversion before the insert.
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
      ContextAwareConversionPatternRewriter& rewriter) const final {
    FailureOr<Attribute> scalarLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getScalar());
    FailureOr<Attribute> destLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getDest());

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

class ConvertTensorExtractSlice
    : public ContextAwareOpConversionPattern<tensor::ExtractSliceOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::ExtractSliceOp>::ContextAwareOpConversionPattern;

  LogicalResult secretSourceSecretResult(
      tensor::ExtractSliceOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const {
    MLIRContext* ctx = op.getContext();

    FailureOr<Attribute> sourceLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getSource());
    FailureOr<Attribute> resultLayoutResult =
        getTypeConverter()->getContextualAttr(op.getResult());
    LayoutAttr resultLayout = cast<LayoutAttr>(resultLayoutResult.value());
    IntegerRelation sourceRel =
        cast<LayoutAttr>(sourceLayoutResult.value()).getIntegerRelation();
    IntegerRelation resultRel = resultLayout.getIntegerRelation();

    // Compute the layout relation of the extract_slice operation.
    auto extractSliceLayout =
        getSliceExtractionRelation(op.getSourceType(), op.getResultType(),
                                   SmallVector<int64_t>(op.getStaticOffsets()),
                                   SmallVector<int64_t>(op.getStaticSizes()),
                                   SmallVector<int64_t>(op.getStaticStrides()));
    if (failed(extractSliceLayout)) {
      return op.emitError() << "failed to get layout for extract slice";
    }

    // Remap the source ciphertext semantic tensor to the result ciphertext
    // semantic tensor layout. To do this, we compose the relations to traverse
    // the following diagram, starting from the source ciphertext tensor to the
    // extracted slice ciphertext tensor.
    //  Source tensor ─────────> Slice tensor
    //     \                        │
    //    /│\                       │
    //     │                       \│ /
    //     │                        \/
    // Source ciphertext        Slice ciphertext
    // (ct, slot)               (ct, slot)
    sourceRel.inverse();
    sourceRel.compose(extractSliceLayout.value());
    sourceRel.compose(resultRel);

    LayoutAttr sliceLayoutAttr =
        LayoutAttr::getFromIntegerRelation(ctx, sourceRel);
    auto resultCiphertextSemanticType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResultType(), resultLayout));
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto remapAndExtract = remapAndExtractResult(
        b, adaptor.getSource(), sliceLayoutAttr, resultCiphertextSemanticType);

    setMaterializedAttr(remapAndExtract);
    setAttributeAssociatedWith(remapAndExtract->getResult(0), kLayoutAttrName,
                               sliceLayoutAttr);
    rewriter.replaceOp(op, remapAndExtract->getResult(0));
    return success();
  }

  LogicalResult matchAndRewrite(
      tensor::ExtractSliceOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
    // If this is already materialized, then we can skip it.
    if (hasMaterializedAttr(op)) return failure();

    // Extract a secret slice from a secret tensor.
    FailureOr<Attribute> sourceLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getSource());
    FailureOr<Attribute> resultLayoutResult =
        getTypeConverter()->getContextualAttr(op.getResult());

    bool isSecretSource = succeeded(sourceLayoutResult);
    bool isSecretResult = succeeded(resultLayoutResult);

    if (isSecretSource && isSecretResult) {
      return secretSourceSecretResult(op, adaptor, rewriter);
    }

    if (isSecretSource && !isSecretResult) {
      return op.emitError()
             << "result tensor should have been assigned a layout "
                "by layout-propagation";
    }

    if (!isSecretSource && isSecretResult) {
      return op.emitError()
             << "source tensor should have been assigned a layout "
                "by layout-propagation";
    }

    // cleartext scalar and cleartext tensor means this is a cleartext op
    // that can be elided.
    setMaterializedAttr(op);
    return success();
  }
};

class ConvertTensorCollapseShape
    : public ContextAwareOpConversionPattern<tensor::CollapseShapeOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::CollapseShapeOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::CollapseShapeOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
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
        getTypeConverter()->getContextualAttr(adaptor.getSrc());
    if (failed(tensorLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    auto tensorLayout = dyn_cast<LayoutAttr>(tensorLayoutResult.value());
    if (!tensorLayout) {
      return op.emitError() << "failed to fetch new layout attribute for input";
    }
    FailureOr<Attribute> resultLayoutResult =
        getTypeConverter()->getContextualAttr(op.getResult());
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(resultLayoutResult.value());
    if (!resultLayout) {
      return op.emitError() << "failed to fetch new layout attribute for input";
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResultType(), resultLayout);

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
    auto castOp = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getSrc());
    setMaterializedAttr(castOp);
    setAttributeAssociatedWith(castOp.getResult(0), kLayoutAttrName,
                               resultLayout);

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

class ConvertTensorExpandShape
    : public ContextAwareOpConversionPattern<tensor::ExpandShapeOp> {
 public:
  using ContextAwareOpConversionPattern<
      tensor::ExpandShapeOp>::ContextAwareOpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::ExpandShapeOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
    // if the layouts are equal (modulo the rank reduction) then this can be a
    // no-op. check if the layouts are equal after collapsing the dimensions.
    // then if the input and output types are the same, just replace with input.
    SliceVerificationResult res =
        isRankReducedType(op.getResultType(), op.getSrcType());
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(
          op, "Only rank-reduced types are supported for CollapseShapeOp");

    FailureOr<Attribute> resultLayoutResult =
        getTypeConverter()->getContextualAttr(op.getResult());
    if (failed(resultLayoutResult)) {
      return op.emitError() << "failed to fetch layout attribute for input";
    }
    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(resultLayoutResult.value());
    if (!resultLayout) {
      return op.emitError() << "failed to fetch new layout attribute for input";
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResultType(), resultLayout);

    auto srcType = adaptor.getSrc().getType();
    FailureOr<Attribute> sourceLayoutResult =
        getTypeConverter()->getContextualAttr(adaptor.getSrc());
    if (failed(sourceLayoutResult)) {
      // If the tensor has no layout, it is a cleartext operation and
      // can be skipped.
      setMaterializedAttr(op);
      return success();
    }
    auto sourceLayout = dyn_cast<LayoutAttr>(sourceLayoutResult.value());
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
    auto castOp = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getSrc());
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

struct ConvertLinalgMatmul
    : public ContextAwareOpConversionPattern<linalg::MatmulOp> {
 public:
  using ContextAwareOpConversionPattern<
      linalg::MatmulOp>::ContextAwareOpConversionPattern;

  ConvertLinalgMatmul(
      const ContextAwareTypeConverter& contextAwareTypeConverter,
      MLIRContext* context)
      : ContextAwareOpConversionPattern(contextAwareTypeConverter, context,
                                        /*benefit=*/10) {}

  LayoutAttr getLayoutAttr(Value value) const {
    auto layoutLookup = getTypeConverter()->getContextualAttr(value);
    if (failed(layoutLookup)) {
      return nullptr;
    }
    return dyn_cast<LayoutAttr>(layoutLookup.value());
  }

  bool supportsBicyclic(linalg::MatmulOp op, OpAdaptor adaptor) const {
    auto kernelAttr = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    return kernelAttr && kernelAttr.getName() == KernelName::MatmulBicyclic;
  }

  void bicyclicKernel(linalg::MatmulOp op, OpAdaptor adaptor,
                      ContextAwareConversionPatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "Converting linalg.matmul op with bicyclic kernel: " << op
               << "\n");

    TypedValue<RankedTensorType> lhs =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[0]);
    SSAValue lhsLeaf(lhs);
    TypedValue<RankedTensorType> rhs =
        cast<TypedValue<RankedTensorType>>(adaptor.getInputs()[1]);
    SSAValue rhsLeaf(rhs);

    auto lhsType = cast<RankedTensorType>(op.getInputs()[0].getType());
    auto rhsType = cast<RankedTensorType>(op.getInputs()[1].getType());

    // TODO(#2368): zero-pad inputs to ensure coprime dimensions
    std::shared_ptr<ArithmeticDagNode<SSAValue>> implementedKernel =
        kernel::implementBicyclicMatmul(lhsLeaf, rhsLeaf, lhsType.getShape()[0],
                                        lhsType.getShape()[1],
                                        rhsType.getShape()[1]);

    rewriter.setInsertionPointAfter(op);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    IRMaterializingVisitor visitor(b, lhs.getType(), [&](Operation* createdOp) {
      setMaterializedAttr(op);
    });
    Value finalOutput = implementedKernel->visit(visitor);

    auto layoutAttr = cast<LayoutAttr>(op->getAttr(kLayoutAttrName));
    auto* finalOutputOp = finalOutput.getDefiningOp();
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
      linalg::MatmulOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const final {
    if (supportsBicyclic(op, adaptor)) {
      bicyclicKernel(op, adaptor, rewriter);
      return success();
    }
    return failure();
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

    patterns.add<ConvertAnyAddingMaterializedAttr, ConvertConvertLayout,
                 ConvertFunc, ConvertLinalgConv2D, ConvertLinalgMatmul,
                 ConvertLinalgMatvecLayout, ConvertLinalgReduce,
                 ConvertSecretGeneric, ConvertTensorCollapseShape,
                 ConvertTensorExpandShape, ConvertTensorExtractLayout,
                 ConvertTensorExtractSlice, ConvertTensorInsertLayout,
                 ConvertTensorInsertSlice>(typeConverter, context);
    patterns.add<ConvertAssignLayout>(typeConverter, context, ciphertextSize);

    ConversionConfig config;
    config.buildMaterializations = false;
    if (failed(applyContextAwarePartialConversion(
            module, target, std::move(patterns), config))) {
      return signalPassFailure();
    }

    // Decompose tensor.concat into repeated tensor.insert_slice ops.
    // Note ConvertAssignLayout generates tensor.concat
    RewritePatternSet cleanupPatterns2(context);
    tensor::populateDecomposeTensorConcatPatterns(cleanupPatterns2);

    // Folding here will remove any unrealized conversion cast ops that were
    // inserted to persist new layouts.
    if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns2)))) {
      return signalPassFailure();
    }

    // Walk the IR to validate that there are no remaining unrealized conversion
    // cast ops.
    module->walk([&](UnrealizedConversionCastOp op) {
      op->emitError() << "unexpected unrealized conversion cast op found";
      signalPassFailure();
    });

    clearAttrs(module, kLayoutAttrName);
    clearAttrs(module, kMaterializedAttrName);
  }
};

}  // namespace heir
}  // namespace mlir
