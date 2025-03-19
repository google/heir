#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using tensor_ext::LayoutAttr;

namespace {

bool containsDim(ArrayRef<int64_t> dims, int64_t dim) {
  return llvm::any_of(dims, [dim](int64_t d) { return d == dim; });
}

Value expandDims(Value value, LayoutAttr layout, ImplicitLocOpBuilder &b,
                 const std::function<void(Operation *)> &createdOpCallback) {
  LLVM_DEBUG(llvm::dbgs() << "Expanding dims...\n");
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();

  // Tensors are handled via tensor.expand_shape
  RankedTensorType dataSemanticType = cast<RankedTensorType>(value.getType());
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
  createdOpCallback(expandOp);
  return expandOp.getResult();
}

Value applyPadding(Value value, LayoutAttr layout, ImplicitLocOpBuilder &b,
                   const std::function<void(Operation *)> &createdOpCallback) {
  LLVM_DEBUG(llvm::dbgs() << "Applying padding...\n");
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

  createdOpCallback(padOp);
  b.setInsertionPointAfter(padOp);
  return padOp.getResult();
}

FailureOr<Value> maybeReplicateAlongAxis(
    tensor_ext::AssignLayoutOp op, Value value, int axis,
    int64_t outputAxisSize, ImplicitLocOpBuilder &b,
    const std::function<void(Operation *)> &createdOpCallback) {
  LLVM_DEBUG(llvm::dbgs() << "Replicating...\n");
  RankedTensorType mostRecentType = cast<RankedTensorType>(value.getType());
  int64_t dataDimSize = mostRecentType.getDimSize(axis);

  if (outputAxisSize % dataDimSize != 0 && dataDimSize % outputAxisSize != 0) {
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
    createdOpCallback(concatOp);
    return concatOp.getResult();
  }
  return value;
}

FailureOr<Value> implementAssignLayoutForTensor(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  RankedTensorType dataSemanticType =
      cast<RankedTensorType>(op.getValue().getType());
  RankedTensorType ciphertextSemanticType = cast<RankedTensorType>(
      materializeLayout(dataSemanticType, op.getLayout(), ciphertextSize));
  LLVM_DEBUG(llvm::dbgs() << "Converting AssignLayoutOp to use result type "
                          << ciphertextSemanticType << "\n");
  Value input = op.getValue();
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
      mostRecentOutput =
          expandDims(mostRecentOutput, layout, builder, createdOpCallback);
    }

    // 2. Add padding to the end of each axis via tensor.pad
    if (alignment.getPadding() && !alignment.getPadding().empty()) {
      mostRecentOutput =
          applyPadding(mostRecentOutput, layout, builder, createdOpCallback);
    }

    // 3. Replicate the input tensor along each axis via tensor.concat
    for (int i = 0; i < alignment.getOut().size(); ++i) {
      FailureOr<Value> res = maybeReplicateAlongAxis(
          op, mostRecentOutput, i, alignment.getOut()[i], builder,
          createdOpCallback);
      if (failed(res)) return res;
      mostRecentOutput = res.value();
    }
  }

  // At this point, we could try to guarantee that the replicated data tensor
  // has the same number of elements as the ciphertext tensor, but in general
  // this is not required. You could just waste slots, though there is a
  // concern that some kernels that rely on replication may not work as
  // expected. So in this case we emit a warning.
  LLVM_DEBUG({
    RankedTensorType mostRecentType =
        cast<RankedTensorType>(mostRecentOutput.getType());
    if (mostRecentType.getNumElements() <
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
    auto emptyOp = builder.create<mlir::arith::ConstantOp>(
        builder.getZeroAttr(ciphertextSemanticType));
    createdOpCallback(emptyOp);

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
    auto materializeLayoutOp = builder.create<linalg::GenericOp>(
        /*resultTypes=*/emptyOp.getResult().getType(),
        /*inputs=*/mostRecentOutput,
        /*outputs=*/emptyOp.getResult(), indexingMaps, iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // Do nothing, which just assigns the input to the output slot.
          auto yieldOp =
              nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
          createdOpCallback(yieldOp);
        });

    createdOpCallback(materializeLayoutOp);
    mostRecentOutput = materializeLayoutOp.getResult(0);
  }

  return mostRecentOutput;
}

FailureOr<Value> implementAssignLayoutForScalar(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  RankedTensorType ciphertextSemanticType =
      cast<RankedTensorType>(materializeScalarLayout(
          op.getResult().getType(), op.getLayout(), ciphertextSize));
  LLVM_DEBUG(
      llvm::dbgs() << "Converting AssignLayoutOp for scalar to use result type "
                   << ciphertextSemanticType << "\n");

  LayoutAttr layout = op.getLayout();
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();
  Value scalar = op.getValue();

  // Common case: no padding, all replication: the entire encoding can be
  // reduced to a single splat.
  if (alignment.getPadding().empty()) {
    auto splatOp =
        builder.create<tensor::SplatOp>(ciphertextSemanticType, scalar);
    createdOpCallback(splatOp);
    return splatOp.getResult();
  }

  // TODO(#1662): improve scalar layout materialization
  return failure();
}

Value implementUnpackOpForTensor(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  RankedTensorType dataSemanticType =
      cast<RankedTensorType>(op.getResult().getType());
  Value input = op.getValue();
  Value mostRecentOutput = input;

  LayoutAttr layout = op.getLayout();
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();
  RankedTensorType replicatedType =
      alignment ? RankedTensorType::get(alignment.getOut(),
                                        dataSemanticType.getElementType())
                : dataSemanticType;

  // 1. Extract the data according to the layout map
  if (!layout.getMap().isIdentity()) {
    // A zero-valued tensor to store the result of the unpacking.
    auto emptyOp = builder.create<mlir::arith::ConstantOp>(
        builder.getZeroAttr(replicatedType));
    createdOpCallback(emptyOp);

    SmallVector<utils::IteratorType> iteratorTypes(
        op.getLayout().getMap().getNumDims(), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps = {
        // The first map corresponds to how the iteration indices map to the
        // ciphertext-semantic tensor's indices. This is the layout map.
        layout.getMap(),
        // The second map maps the iteration indices to the data-semantic
        // tensor's indices.
        AffineMap::getMultiDimIdentityMap(layout.getMap().getNumDims(),
                                          op.getContext()),
    };
    auto inverseLayoutOp = builder.create<linalg::GenericOp>(
        /*resultTypes=*/emptyOp.getResult().getType(),
        /*inputs=*/mostRecentOutput,
        /*outputs=*/emptyOp.getResult(), indexingMaps, iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // Do nothing, which just assigns the input to the output slot.
          auto yieldOp =
              nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
          createdOpCallback(yieldOp);
        });
    mostRecentOutput = inverseLayoutOp.getResult(0);
  }

  if (alignment) {
    // 2. Slice the input to undo repetition and padding.
    // Because the repetition and padding only apply to the tail end of each
    // axis, we can just slice the tensor on the leading part of each axis to
    // construct the original (inserted-dim)-sized tensor.
    DenseMap<int64_t, int64_t> inAxisToReplicatedAxis;
    SmallVector<OpFoldResult> offsets(replicatedType.getRank(),
                                      builder.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(replicatedType.getRank(),
                                      builder.getIndexAttr(1));
    ArrayRef<int64_t> insertedDims = alignment.getInsertedDims().asArrayRef();
    SmallVector<OpFoldResult> sizes;
    int inDim = 0;
    for (size_t i = 0; i < replicatedType.getRank(); ++i) {
      if (std::find(insertedDims.begin(), insertedDims.end(), i) !=
          insertedDims.end()) {
        // This is an inserted dimension, so it becomes a unit axis and can be
        // dropped.
        sizes.push_back(builder.getIndexAttr(1));
      } else {
        // This is an original dimension, so slice it by the original size of
        // the data semantic tensor.
        sizes.push_back(builder.getIndexAttr(alignment.getIn()[inDim]));
        inDim++;
      }
    }

    auto extractSliceOp = builder.create<tensor::ExtractSliceOp>(
        dataSemanticType, mostRecentOutput, offsets, sizes, strides);
    createdOpCallback(extractSliceOp);
    mostRecentOutput = extractSliceOp.getResult();
  }

  return mostRecentOutput;
}

Value implementUnpackOpForScalar(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  // All we need to do here is determine the index to extract from
  SmallVector<Value> indices;
  LayoutAttr layout = op.getLayout();
  tensor_ext::AlignmentAttr alignment = layout.getAlignment();

  // Note padding only inserts at the end of a tensor axis, so regardless
  // of any repetition, the first entry of the tensor will always contain
  // the right value. So all we need to do is insert a constant zero for
  // each dimension of the input tensor.
  for (unsigned i = 0; i < alignment.getOut().size(); ++i) {
    // We need to insert a 0 for each inserted dimension.
    auto constOp = builder.create<arith::ConstantIndexOp>(0);
    createdOpCallback(constOp);
    indices.push_back(constOp);
  }

  auto splatOp = builder.create<tensor::ExtractOp>(op.getResult().getType(),
                                                   op.getValue(), indices);
  createdOpCallback(splatOp);
  return splatOp.getResult();
}

}  // namespace

FailureOr<Value> implementAssignLayout(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  if (isa<RankedTensorType>(op.getResult().getType())) {
    return implementAssignLayoutForTensor(op, ciphertextSize, builder,
                                          createdOpCallback);
  }

  return implementAssignLayoutForScalar(op, ciphertextSize, builder,
                                        createdOpCallback);
};

Value implementUnpackOp(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder &builder,
    const std::function<void(Operation *)> &createdOpCallback) {
  if (isa<RankedTensorType>(op.getResult().getType())) {
    return implementUnpackOpForTensor(op, builder, createdOpCallback);
  }

  return implementUnpackOpForScalar(op, builder, createdOpCallback);
}

}  // namespace heir
}  // namespace mlir
