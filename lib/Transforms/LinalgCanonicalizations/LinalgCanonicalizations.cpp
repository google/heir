#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallBitVector.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"          // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "linalg-canonicalizations"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LINALGCANONICALIZATIONS
#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h.inc"

struct FoldBroadcastExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
 public:
  FoldBroadcastExtractSlice(MLIRContext* context)
      : OpRewritePattern<tensor::ExtractSliceOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter& rewriter) const override {
    auto broadcastOp =
        extractSliceOp.getSource().getDefiningOp<linalg::BroadcastOp>();
    if (!broadcastOp) return failure();

    auto defConstantOp =
        broadcastOp.getInput().getDefiningOp<arith::ConstantOp>();
    if (!defConstantOp) return failure();

    ElementsAttr elementsAttr =
        dyn_cast<ElementsAttr>(defConstantOp.getValueAttr());
    if (auto denseResourceAttr =
            dyn_cast<DenseResourceElementsAttr>(defConstantOp.getValueAttr())) {
      const auto data = denseResourceAttr.getData();
      elementsAttr = DenseElementsAttr::getFromRawBuffer(
          denseResourceAttr.getType(), data);
    }

    if (!elementsAttr) return failure();

    auto isAllConstant = [](ArrayRef<OpFoldResult> mixed) {
      return llvm::all_of(mixed,
                          [](OpFoldResult ofr) { return isa<Attribute>(ofr); });
    };
    if (!isAllConstant(extractSliceOp.getMixedOffsets()) ||
        !isAllConstant(extractSliceOp.getMixedSizes()) ||
        !isAllConstant(extractSliceOp.getMixedStrides()))
      return failure();

    auto resultType = extractSliceOp.getType();
    if (!resultType.hasStaticShape()) return failure();

    auto broadcastInType =
        cast<RankedTensorType>(broadcastOp.getInput().getType());
    auto broadcastOutType =
        cast<RankedTensorType>(broadcastOp.getInit().getType());
    auto broadcastDims = broadcastOp.getDimensions();

    int64_t broadcastOutRank = broadcastOutType.getRank();
    SmallVector<int64_t> outToInDim(broadcastOutRank, -1);
    llvm::DenseSet<int64_t> addedDimsSet(broadcastDims.begin(),
                                         broadcastDims.end());
    int64_t inDimIdx = 0;
    for (int64_t i = 0; i < broadcastOutRank; ++i) {
      if (addedDimsSet.find(i) == addedDimsSet.end()) {
        outToInDim[i] = inDimIdx++;
      }
    }

    auto offsets = extractSliceOp.getStaticOffsets();
    auto strides = extractSliceOp.getStaticStrides();
    auto droppedDims = extractSliceOp.getDroppedDims();

    int64_t numElements = resultType.getNumElements();
    auto resultShape = resultType.getShape();

    auto inputValues = elementsAttr.getValues<Attribute>();

    SmallVector<Attribute> resultValues;
    resultValues.reserve(numElements);

    for (int64_t i = 0; i < numElements; ++i) {
      auto resultIndices = getIndicesFromRowMajorShape(
          i, SmallVector<int64_t>(resultShape.begin(), resultShape.end()));

      SmallVector<int64_t> fullSourceIndices(broadcastOutRank);
      int64_t rDim = 0;
      for (int64_t sDim = 0; sDim < broadcastOutRank; ++sDim) {
        if (droppedDims.test(sDim)) {
          fullSourceIndices[sDim] = offsets[sDim];
        } else {
          fullSourceIndices[sDim] =
              offsets[sDim] + resultIndices[rDim++] * strides[sDim];
        }
      }

      SmallVector<int64_t> inputIndices;
      for (int64_t sDim = 0; sDim < broadcastOutRank; ++sDim) {
        if (outToInDim[sDim] != -1) {
          inputIndices.push_back(fullSourceIndices[sDim]);
        }
      }

      auto flatIndex = getFlattenedIndex(
          broadcastInType,
          llvm::map_to_vector(inputIndices, [&](int64_t idx) -> OpFoldResult {
            return rewriter.getIndexAttr(idx);
          }));
      if (failed(flatIndex)) return failure();

      resultValues.push_back(inputValues[flatIndex.value()]);
    }

    auto resultAttr = DenseElementsAttr::get(resultType, resultValues);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(extractSliceOp, resultAttr);
    return success();
  }
};

struct FoldConstantLinalgTranspose
    : public OpRewritePattern<mlir::linalg::TransposeOp> {
 public:
  FoldConstantLinalgTranspose(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::TransposeOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::TransposeOp transposeOp,
                                PatternRewriter& rewriter) const override {
    ArrayRef<int64_t> outputShape = transposeOp.getInit().getType().getShape();
    ArrayRef<int64_t> permutation = transposeOp.getPermutation();

    TypedAttr attr;
    if (!matchPattern(transposeOp.getInput(), m_Constant(&attr)))
      return rewriter.notifyMatchFailure(transposeOp,
                                         "transpose input must be a constant");
    DenseElementsAttr denseAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!denseAttr)
      return rewriter.notifyMatchFailure(
          transposeOp, "constant input must be a dense elements attribute");

    auto elementType = denseAttr.getElementType();
    if (elementType.isIntOrIndexOrFloat()) {
      DenseElementsAttr transposedDenseElementsAttr;
      if (auto intType = dyn_cast<IntegerType>(elementType)) {
        // This switch statement handles the different integer types with
        // templates instead of only using a generic APInt so that the
        // performance of accessing and transposing the element values is
        // improved.
        switch (intType.getWidth()) {
          case 1:
            transposedDenseElementsAttr =
                convertConstantToTransposedElementsAttr<bool>(
                    denseAttr, outputShape, permutation);
            break;
          case 8:
            transposedDenseElementsAttr =
                convertConstantToTransposedElementsAttr<int8_t>(
                    denseAttr, outputShape, permutation);
            break;
          case 16:
            transposedDenseElementsAttr =
                convertConstantToTransposedElementsAttr<int16_t>(
                    denseAttr, outputShape, permutation);
            break;
          case 32:
            transposedDenseElementsAttr =
                convertConstantToTransposedElementsAttr<int32_t>(
                    denseAttr, outputShape, permutation);
            break;
          case 64:
            transposedDenseElementsAttr =
                convertConstantToTransposedElementsAttr<int64_t>(
                    denseAttr, outputShape, permutation);
            break;
          default:
            transposedDenseElementsAttr =
                convertConstantToTransposedElementsAttr<APInt>(
                    denseAttr, outputShape, permutation);
        }
      } else {  // floating point
        // Likewise, floating point types are handled with templates to enable
        // better performance.
        if (elementType.isF32()) {
          transposedDenseElementsAttr =
              convertConstantToTransposedElementsAttr<float>(
                  denseAttr, outputShape, permutation);
        } else if (elementType.isF64()) {
          transposedDenseElementsAttr =
              convertConstantToTransposedElementsAttr<double>(
                  denseAttr, outputShape, permutation);
        } else {
          transposedDenseElementsAttr =
              convertConstantToTransposedElementsAttr<APFloat>(
                  denseAttr, outputShape, permutation);
        }
      }
      auto transposedConstantOp = arith::ConstantOp::create(
          rewriter, transposeOp.getLoc(), transposedDenseElementsAttr);
      rewriter.replaceOp(transposeOp, transposedConstantOp.getResult());
      return success();
    }
    return rewriter.notifyMatchFailure(transposeOp, "unsupported element type");
  }

 private:
  // Converts a constant DenseElementsAttr to a transposed DenseElementsAttr.
  template <typename BaseType>
  DenseElementsAttr convertConstantToTransposedElementsAttr(
      DenseElementsAttr inputDenseAttr, ArrayRef<int64_t> outputShape,
      ArrayRef<int64_t> permutation) const {
    auto attrValues = inputDenseAttr.getValues<BaseType>();
    auto numElements = inputDenseAttr.getNumElements();
    auto inputShape = inputDenseAttr.getType().getShape();
    SmallVector<BaseType> transposedValues;
    transposedValues.reserve(numElements);

    // For each element in the transposed tensor, find the corresponding element
    // in the original tensor, and add it to the vector of transposed values.
    for (int i = 0; i < numElements; ++i) {
      int64_t originalIndex = convertTransposedIndexToOriginalIndex(
          inputShape, numElements, permutation, i);
      transposedValues.push_back(attrValues[originalIndex]);
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(outputShape, inputDenseAttr.getElementType()),
        llvm::ArrayRef<BaseType>(transposedValues));
  }

  // Converts a transposed index to an original index.
  int64_t convertTransposedIndexToOriginalIndex(ArrayRef<int64_t> inputShape,
                                                int64_t totalSize,
                                                ArrayRef<int64_t> permutation,
                                                int64_t transposedIndex) const {
    // First compute the position of the transposed index.
    int64_t remainder = transposedIndex;
    int64_t remainingTensorSize = totalSize;
    int64_t inputShapeSize = static_cast<int64_t>(inputShape.size());
    SmallVector<int64_t> transposedIndices;
    transposedIndices.reserve(inputShape.size());
    for (int64_t dim = 0; dim < inputShapeSize; ++dim) {
      int64_t dimSize = inputShape[permutation[dim]];
      remainingTensorSize /= dimSize;
      int64_t indexAtDim = remainder / remainingTensorSize;
      remainder %= remainingTensorSize;
      transposedIndices.push_back(indexAtDim);
    }

    // Convert to the original indices by applying the permutation.
    SmallVector<int64_t> originalIndices(permutation.size(), 0);
    for (int64_t dim = 0; dim < inputShapeSize; ++dim) {
      originalIndices[permutation[dim]] = transposedIndices[dim];
    }

    // Then compute the index in the original tensor.
    int64_t originalIndex = 0;
    int64_t remainingTransposedTensorSize = totalSize;
    for (int64_t dim = 0; dim < inputShapeSize; ++dim) {
      remainingTransposedTensorSize /= inputShape[dim];
      originalIndex += originalIndices[dim] * remainingTransposedTensorSize;
    }
    return originalIndex;
  }
};

struct FoldConstantFill : public OpRewritePattern<mlir::linalg::FillOp> {
 public:
  FoldConstantFill(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::FillOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::FillOp fillOp,
                                PatternRewriter& rewriter) const override {
    auto value = getAsOpFoldResult(fillOp.getInputs()[0]);
    if (isa<Value>(value))
      return rewriter.notifyMatchFailure(fillOp,
                                         "fill value must be a constant");
    if (fillOp.getResults().empty()) {
      // memref semantics
      return rewriter.notifyMatchFailure(
          fillOp, "fillOp with memref semantics not supported");
    }
    auto outputTy = cast<RankedTensorType>(fillOp.getResultTypes()[0]);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        fillOp, outputTy,
        DenseElementsAttr::get(outputTy, cast<Attribute>(value)));
    return success();
  }
};

struct FoldConstantBroadcast
    : public OpRewritePattern<mlir::linalg::BroadcastOp> {
 public:
  FoldConstantBroadcast(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::BroadcastOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::BroadcastOp broadcastOp,
                                PatternRewriter& rewriter) const override {
    auto value = getAsOpFoldResult(broadcastOp.getInput());
    if (isa<Value>(value))
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "broadcast input must be a constant");

    auto elementsAttr = dyn_cast<ElementsAttr>(cast<Attribute>(value));
    if (auto denseResourceAttr =
            dyn_cast<DenseResourceElementsAttr>(cast<Attribute>(value))) {
      const auto data = denseResourceAttr.getData();
      // Limit size to avoid huge constants in IR.
      if (data.size() > 1024 * 1024)
        return rewriter.notifyMatchFailure(
            broadcastOp, "broadcast output too large to fold");

      elementsAttr = DenseElementsAttr::getFromRawBuffer(
          denseResourceAttr.getType(), data);
    }

    if (!elementsAttr)
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "input must be an elements attribute");

    auto outputTy = cast<RankedTensorType>(broadcastOp.getResultTypes()[0]);

    // Replace with the new broadcasted constant.
    // If it's a splat, we can stay with splat.
    if (auto splatInput = dyn_cast<SplatElementsAttr>(elementsAttr)) {
      auto broadcastedInput = SplatElementsAttr::get(
          outputTy, splatInput.getSplatValue<Attribute>());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(broadcastOp, outputTy,
                                                     broadcastedInput);
      return success();
    }

    auto broadcastInType =
        cast<RankedTensorType>(broadcastOp.getInput().getType());
    auto broadcastDims = broadcastOp.getDimensions();
    int64_t broadcastOutRank = outputTy.getRank();

    SmallVector<int64_t> outToInDim(broadcastOutRank, -1);
    llvm::DenseSet<int64_t> addedDimsSet(broadcastDims.begin(),
                                         broadcastDims.end());
    int64_t inDimIdx = 0;
    for (int64_t i = 0; i < broadcastOutRank; ++i) {
      if (addedDimsSet.find(i) == addedDimsSet.end()) {
        outToInDim[i] = inDimIdx++;
      }
    }

    auto inputValues = elementsAttr.getValues<Attribute>();
    SmallVector<Attribute> resultValues;
    int64_t numElements = outputTy.getNumElements();
    resultValues.reserve(numElements);

    auto resultShape = outputTy.getShape();
    for (int64_t i = 0; i < numElements; ++i) {
      auto resultIndices = getIndicesFromRowMajorShape(
          i, SmallVector<int64_t>(resultShape.begin(), resultShape.end()));
      SmallVector<int64_t> inputIndices;
      for (int64_t sDim = 0; sDim < broadcastOutRank; ++sDim) {
        if (outToInDim[sDim] != -1) {
          inputIndices.push_back(resultIndices[sDim]);
        }
      }
      auto flatIndex = getFlattenedIndex(
          broadcastInType,
          llvm::map_to_vector(inputIndices, [&](int64_t idx) -> OpFoldResult {
            return rewriter.getIndexAttr(idx);
          }));
      if (failed(flatIndex)) return failure();
      resultValues.push_back(inputValues[flatIndex.value()]);
    }

    auto resultAttr = DenseElementsAttr::get(outputTy, resultValues);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(broadcastOp, outputTy,
                                                   resultAttr);
    return success();
  }
};

// Folds linalg.map operations that apply a single elementwise operation into
// the elementwise operation on the tensors. This requires that the destination
// tensor was created with a tensor.empty operation.
struct LinalgMapToElementwise : public OpRewritePattern<mlir::linalg::MapOp> {
 public:
  LinalgMapToElementwise(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::MapOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::MapOp mapOp,
                                PatternRewriter& rewriter) const override {
    auto dest = mapOp.getInit();
    // The mapper should have exactly two operations (the second is a yield).
    auto* mapper = mapOp.getBody(0);
    if (mapper->getOperations().size() != 2)
      return rewriter.notifyMatchFailure(
          mapOp, "mapper block must have exactly two operations");

    // The operation should be elementwise.
    Operation& op = mapper->getOperations().front();
    if (!op.hasTrait<mlir::OpTrait::Elementwise>())
      return rewriter.notifyMatchFailure(
          mapOp, "operation in mapper must be elementwise");

    auto* elementwiseOp = rewriter.create(
        mapOp->getLoc(), op.getName().getIdentifier(), mapOp.getInputs(),
        TypeRange(dest.getType()), mapOp->getAttrs(), {}, {});
    rewriter.replaceOp(mapOp, elementwiseOp);
    if (dest.use_empty() && dest.getDefiningOp())
      rewriter.eraseOp(dest.getDefiningOp());
    return success();
  }
};

// Folds linalg.generic operations that apply a elementwise operations
// into the elementwise operations on the tensors. The indexing maps must all be
// identity maps and the iterator types must all be parallel.
struct LinalgGenericToElementwise
    : public OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  LinalgGenericToElementwise(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::GenericOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    // The operations should be elementwise or constants.
    for (auto& op : genericOp.getBody()->getOperations()) {
      if (isa<linalg::YieldOp>(op) || isa<arith::ConstantOp>(op)) continue;
      if (!OpTrait::hasElementwiseMappableTraits(&op)) {
        return rewriter.notifyMatchFailure(
            genericOp,
            "requires operations in generic body to have elementwise "
            "mappable traits");
      }
    }

    // Check that the indexing maps are the identity.
    for (auto mapAttr : genericOp.getIndexingMapsArray()) {
      if (!mapAttr.isIdentity()) {
        return rewriter.notifyMatchFailure(
            genericOp,
            "genericOp with non-identity indexing map not supported");
      }
    }

    // Check that the iterator types are all parallel.
    for (auto iteratorType : genericOp.getIteratorTypesArray()) {
      if (iteratorType != utils::IteratorType::parallel) {
        return rewriter.notifyMatchFailure(
            genericOp,
            "genericOp with non-parallel iterator type not supported");
      }
    }

    // Now replace the body with corresponding elementwise operations.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(genericOp);
    IRMapping bvm;

    // Map the block arguments to the tensor operands of the generic op.
    for (auto const& [i, arg] :
         llvm::enumerate(genericOp.getRegion().getArguments())) {
      bvm.map(arg, genericOp.getOperand(i));
    }

    for (auto& innerOp : genericOp.getBody()->getOperations()) {
      if (isa<linalg::YieldOp>(innerOp)) {
        auto results = llvm::to_vector(llvm::map_range(
            innerOp.getOperands(), [&](Value v) { return bvm.lookup(v); }));
        rewriter.replaceOp(genericOp, results);
        break;
      }

      // Since the generic indexing map and iterator types are the same for all
      // results, each result shape should be the same so use the first result
      // to infer the shapes of the tensor values.
      auto resultTensorShape =
          cast<RankedTensorType>(genericOp.getResult(0).getType()).getShape();
      for (auto& operand : innerOp.getOpOperands()) {
        Value operandValue = operand.get();
        if (auto blockArgument = dyn_cast<BlockArgument>(operandValue)) {
          assert(blockArgument.getArgNumber() < genericOp.getInputs().size() &&
                 "expecting that the elementwise operation does not use the "
                 "output's initial values");
          // Map the block argument to the operand of the generic op.
          bvm.map(operandValue,
                  genericOp.getOperand(blockArgument.getArgNumber()));
        } else if (operandValue.getParentBlock() != genericOp.getBody()) {
          // If this isn't a block argument and wasn't defined in the generic,
          // then it must be a constant from outside the generic. The shape of
          // the splat constant must match the result of this op since
          // all of the operations are elementwise.
          Value splatTensor = tensor::SplatOp::create(
              rewriter, genericOp.getLoc(),
              RankedTensorType::get(resultTensorShape, operandValue.getType()),
              operandValue);
          bvm.map(operandValue, splatTensor);
        } else {
          // This was defined within the generic op so it must be a result of
          // some other innerOp and its value must have a mapping in bvm.
          assert(bvm.lookup(operandValue) &&
                 "expected a mapping for this value");
        }
      }
      auto newInputs = llvm::to_vector(llvm::map_range(
          innerOp.getOperands(), [&](Value v) { return bvm.lookup(v); }));
      auto newDestTypes = llvm::to_vector(
          llvm::map_range(innerOp.getResultTypes(), [&](Type t) -> Type {
            return RankedTensorType::get(resultTensorShape, t);
          }));
      SmallVector<NamedAttribute> newAttrs = {innerOp.getAttrs().begin(),
                                              innerOp.getAttrs().end()};
      for (auto attr : genericOp->getAttrs()) {
        if (attr.getName().getValue() != "iterator_types" &&
            attr.getName().getValue() != "indexing_maps" &&
            attr.getName().getValue() != "operandSegmentSizes") {
          newAttrs.push_back(attr);
        }
      }
      auto* elementwiseOp = rewriter.create(
          genericOp->getLoc(), innerOp.getName().getIdentifier(), newInputs,
          newDestTypes, newAttrs, {}, {});
      bvm.map(innerOp.getResults(), elementwiseOp->getResults());
    }

    return success();
  }
};

struct BroadcastToExpandShape
    : public OpRewritePattern<mlir::linalg::BroadcastOp> {
 public:
  BroadcastToExpandShape(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::BroadcastOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::BroadcastOp broadcastOp,
                                PatternRewriter& rewriter) const override {
    // If the broadcast is only reassociating dimensions, replace with
    // expand_shape.
    SliceVerificationResult res = isRankReducedType(
        broadcastOp.getInit().getType(), broadcastOp.getInput().getType());
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(
          broadcastOp, "broadcast must be a rank-reducing broadcast");

    SmallVector<ReassociationIndices> expandingMap =
        getReassociationForReshapeAtDim(
            broadcastOp.getInit().getType().getRank(),
            broadcastOp.getDimensions());
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        broadcastOp, broadcastOp.getInit().getType(), broadcastOp.getInput(),
        expandingMap);
    return success();
  }
};

struct RewriteTransposedVecmat
    : public OpRewritePattern<mlir::linalg::VecmatOp> {
 public:
  RewriteTransposedVecmat(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::VecmatOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::VecmatOp vecmatOp,
                                PatternRewriter& rewriter) const override {
    auto transposeOp =
        vecmatOp.getInputs()[1].getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp)
      return rewriter.notifyMatchFailure(
          vecmatOp, "second input to vecmat must be a transpose op");

    rewriter.replaceOpWithNewOp<linalg::MatvecOp>(
        vecmatOp, vecmatOp.getResultTypes()[0],
        ValueRange{transposeOp.getInput(), vecmatOp.getInputs()[0]},
        vecmatOp.getDpsInits()[0]);
    return success();
  }
};

struct RewriteTransposedMatvec
    : public OpRewritePattern<mlir::linalg::MatvecOp> {
 public:
  RewriteTransposedMatvec(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::MatvecOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::MatvecOp matvecOp,
                                PatternRewriter& rewriter) const override {
    auto transposeOp =
        matvecOp.getInputs()[0].getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp)
      return rewriter.notifyMatchFailure(
          matvecOp, "first input to matvec must be a transpose op");

    rewriter.replaceOpWithNewOp<linalg::VecmatOp>(
        matvecOp, matvecOp.getResultTypes()[0],
        ValueRange{matvecOp.getInputs()[1], transposeOp.getInput()},
        matvecOp.getDpsInits()[0]);
    return success();
  }
};

struct RewriteAvgPoolAsConv1D
    : public OpRewritePattern<mlir::linalg::PoolingNcwSumOp> {
 public:
  RewriteAvgPoolAsConv1D(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::PoolingNcwSumOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::PoolingNcwSumOp poolOp,
                                PatternRewriter& rewriter) const override {
    auto inputTy = cast<RankedTensorType>(poolOp.getInputs()[0].getType());
    auto filterTy = cast<RankedTensorType>(poolOp.getInputs()[1].getType());
    auto outputTy = cast<RankedTensorType>(poolOp.getResultTypes()[0]);

    auto c = inputTy.getDimSize(1);
    auto eltTy = filterTy.getElementType();
    auto kernelShape = SmallVector<int64_t>{c, c, filterTy.getDimSize(0)};
    auto kernelTy = RankedTensorType::get(kernelShape, eltTy);

    // Create kernel value attributes of ones and zeros for the filter.
    Attribute zeroAttr = rewriter.getZeroAttr(eltTy);
    Attribute oneAttr = rewriter.getOneAttr(eltTy);

    // If there is a constant division following the sum pool, update the
    // kernel of ones to be 1 / divValue. This is a common enough pattern since
    // it represents an average pool.
    Attribute avgAttr = oneAttr;
    Value avgPoolOutput;
    if (poolOp->hasOneUse()) {
      auto divOp = dyn_cast<arith::DivFOp>(*poolOp->getUsers().begin());
      if (divOp) {
        OpOperand& use = *poolOp->getUses().begin();
        if (auto constantAttr = dyn_cast<Attribute>(getAsOpFoldResult(
                divOp->getOperand(1 - use.getOperandNumber())))) {
          if (auto splatAttr = dyn_cast<SplatElementsAttr>(
                  cast<DenseElementsAttr>(constantAttr))) {
            auto divValue = splatAttr.getSplatValue<APFloat>();
            APFloat one = APFloat::getOne(divValue.getSemantics());
            avgAttr = rewriter.getFloatAttr(eltTy, one / divValue);
            avgPoolOutput = divOp.getResult();
          }
        }
      }
    }

    // Build average pooling kernel as a special type of convolution. The kernel
    // computes a window average, so it is a fixed constant
    // (1 / divValue) where f == c and zeros where f != c (so each
    // channel is averaged independently) and strides equal to the pooling
    // sizes. See
    // https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
    int64_t w = filterTy.getDimSize(0);
    int64_t numElements = c * c * w;
    SmallVector<Attribute> values(numElements, zeroAttr);

    for (int64_t f_idx = 0; f_idx < c; ++f_idx) {
      for (int64_t c_idx = 0; c_idx < c; ++c_idx) {
        if (f_idx == c_idx) {
          for (int64_t w_idx = 0; w_idx < w; ++w_idx) {
            int64_t idx = f_idx * (c * w) + c_idx * w + w_idx;
            values[idx] = avgAttr;
          }
        }
      }
    }

    TypedAttr kernelVals = DenseElementsAttr::get(kernelTy, values);
    auto kernel =
        arith::ConstantOp::create(rewriter, poolOp.getLoc(), kernelVals);
    Value conv = linalg::Conv1DNcwFcwOp::create(
                     rewriter, poolOp.getLoc(), outputTy,
                     ValueRange{poolOp.getInputs()[0], kernel},
                     ValueRange{poolOp.getOutputs()[0]}, poolOp.getStrides(),
                     poolOp.getDilations())
                     .getResult(0);

    if (avgPoolOutput) {
      rewriter.replaceAllUsesWith(avgPoolOutput, conv);
    } else {
      rewriter.replaceOp(poolOp, conv);
    }
    return success();
  }
};

struct RewriteAvgPoolAsConv2D
    : public OpRewritePattern<mlir::linalg::PoolingNchwSumOp> {
 public:
  RewriteAvgPoolAsConv2D(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::PoolingNchwSumOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::PoolingNchwSumOp poolOp,
                                PatternRewriter& rewriter) const override {
    auto inputTy = cast<RankedTensorType>(poolOp.getInputs()[0].getType());
    auto filterTy = cast<RankedTensorType>(poolOp.getInputs()[1].getType());
    auto outputTy = cast<RankedTensorType>(poolOp.getResultTypes()[0]);

    auto c = inputTy.getDimSize(1);
    auto eltTy = filterTy.getElementType();
    auto kernelShape = SmallVector<int64_t>{c, c, filterTy.getDimSize(0),
                                            filterTy.getDimSize(1)};
    auto kernelTy = RankedTensorType::get(kernelShape, eltTy);

    // Create kernel value attributes of ones and zeros for the filter.
    Attribute zeroAttr = rewriter.getZeroAttr(eltTy);
    Attribute oneAttr = rewriter.getOneAttr(eltTy);

    // If there is a constant division following the sum pool, update the
    // kernel of ones to be 1 / divValue. This is a common enough pattern since
    // it represents an average pool.
    Attribute avgAttr = oneAttr;
    Value avgPoolOutput;
    if (poolOp->hasOneUse()) {
      auto divOp = dyn_cast<arith::DivFOp>(*poolOp->getUsers().begin());
      if (divOp) {
        OpOperand& use = *poolOp->getUses().begin();
        if (auto constantAttr = dyn_cast<Attribute>(getAsOpFoldResult(
                divOp->getOperand(1 - use.getOperandNumber())))) {
          if (auto splatAttr = dyn_cast<SplatElementsAttr>(
                  cast<DenseElementsAttr>(constantAttr))) {
            auto divValue = splatAttr.getSplatValue<APFloat>();
            APFloat one = APFloat::getOne(divValue.getSemantics());
            avgAttr = rewriter.getFloatAttr(eltTy, one / divValue);
            avgPoolOutput = divOp.getResult();
          }
        }
      }
    }

    // Build average pooling kernel as a special type of convolution. The kernel
    // computes an average of pixels in a zone, so it constants a fixed constant
    // (one or 1 / divValue) where f == c and zeros where f != c (so each
    // channel is averaged independently) and strides equal to the pooling
    // sizes. See
    // https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
    int64_t kh = filterTy.getDimSize(0);
    int64_t kw = filterTy.getDimSize(1);
    int64_t numElements = c * c * kh * kw;
    SmallVector<Attribute> values(numElements, zeroAttr);

    for (int64_t f_idx = 0; f_idx < c; ++f_idx) {
      for (int64_t c_idx = 0; c_idx < c; ++c_idx) {
        if (f_idx == c_idx) {
          for (int64_t h_idx = 0; h_idx < kh; ++h_idx) {
            for (int64_t w_idx = 0; w_idx < kw; ++w_idx) {
              int64_t idx = f_idx * (c * kh * kw) + c_idx * (kh * kw) +
                            h_idx * kw + w_idx;
              values[idx] = avgAttr;
            }
          }
        }
      }
    }

    TypedAttr kernelVals = DenseElementsAttr::get(kernelTy, values);
    auto kernel =
        arith::ConstantOp::create(rewriter, poolOp.getLoc(), kernelVals);
    Value conv = linalg::Conv2DNchwFchwOp::create(
                     rewriter, poolOp.getLoc(), outputTy,
                     ValueRange{poolOp.getInputs()[0], kernel},
                     ValueRange{poolOp.getOutputs()[0]}, poolOp.getStrides(),
                     poolOp.getDilations())
                     .getResult(0);

    if (avgPoolOutput) {
      rewriter.replaceAllUsesWith(avgPoolOutput, conv);
    } else {
      rewriter.replaceOp(poolOp, conv);
    }
    return success();
  }
};

static SmallVector<int64_t> getBroadcastDimensions(AffineMap map,
                                                   int64_t numDims) {
  llvm::SmallDenseSet<unsigned> usedDims;
  for (auto expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      usedDims.insert(dimExpr.getPosition());
    }
  }
  SmallVector<int64_t> addedDims;
  for (int i = 0; i < numDims; ++i) {
    if (usedDims.find(i) == usedDims.end()) {
      addedDims.push_back(i);
    }
  }
  return addedDims;
}

/// A rewrite pattern that materializes broadcasts for broadcasting operands in
/// linalg.generic ops with parallel iterators.
///
/// This pattern matches linalg.generic ops where all iterator types are
/// parallel, and at least one operand has a broadcasting indexing map (i.e.,
/// the map drops dimensions, mapping a larger iteration space to a smaller
/// operand space). It creates a linalg.broadcast op for each such operand to
/// materialize the broadcast, making the operand match the output shape.
/// This allows subsequent patterns (like LinalgGenericToElementwise) to convert
/// the op to elementwise operations.
struct MaterializeBroadcasts : public OpRewritePattern<linalg::GenericOp> {
 public:
  MaterializeBroadcasts(MLIRContext* context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    // Only handle ops with multiple inputs to avoid infinite loops when
    // materializing broadcasts. Single-input broadcast ops don't need to be
    // converted to elementwise ops.
    if (genericOp.getNumDpsInputs() <= 1) return failure();

    for (auto iteratorType : genericOp.getIteratorTypesArray()) {
      if (iteratorType != utils::IteratorType::parallel) {
        return failure();
      }
    }

    auto indexingMaps = genericOp.getIndexingMapsArray();
    bool madeChanges = false;
    SmallVector<Value> newInputs;
    SmallVector<AffineMap> newMaps;

    int64_t numDims = genericOp.getNumLoops();

    for (int64_t i = 0; i < genericOp.getNumDpsInputs(); ++i) {
      OpOperand* operand = genericOp.getDpsInputOperand(i);
      AffineMap map = indexingMaps[i];
      Value value = operand->get();

      if (map.isIdentity()) {
        newInputs.push_back(value);
        newMaps.push_back(map);
        continue;
      }

      if (map.getNumResults() < numDims) {
        madeChanges = true;
        auto materializedValue = materializeBroadcastForOperand(
            rewriter, genericOp, value, map, numDims);
        if (failed(materializedValue)) {
          return failure();
        }
        newInputs.push_back(*materializedValue);
        newMaps.push_back(rewriter.getMultiDimIdentityMap(numDims));
      } else {
        newInputs.push_back(value);
        newMaps.push_back(map);
      }
    }

    if (!madeChanges) return failure();

    for (int64_t i = 0; i < genericOp.getNumDpsInits(); ++i) {
      newMaps.push_back(indexingMaps[genericOp.getNumDpsInputs() + i]);
    }

    auto newGenericOp = linalg::GenericOp::create(
        rewriter, genericOp.getLoc(), genericOp.getResultTypes(), newInputs,
        genericOp.getDpsInits(), newMaps, genericOp.getIteratorTypesArray());

    rewriter.inlineRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                                newGenericOp.getRegion().begin());
    rewriter.replaceOp(genericOp, newGenericOp.getResults());

    return success();
  }

 private:
  LogicalResult tryCollapseUnitDims(PatternRewriter& rewriter, Location loc,
                                    Value& value, AffineMap& map,
                                    int64_t targetRank) const {
    auto inputType = cast<RankedTensorType>(value.getType());
    if (inputType.getRank() <= targetRank) return success();

    SmallVector<int64_t> dimsToDrop;
    for (unsigned j = 0; j < map.getNumResults(); ++j) {
      auto expr = map.getResult(j);
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        if (inputType.getShape()[j] == 1) {
          dimsToDrop.push_back(j);
        }
      }
    }

    int64_t newRank = inputType.getRank() - dimsToDrop.size();
    if (dimsToDrop.empty() || newRank > targetRank) {
      return failure();
    }

    auto reassociation =
        getReassociationForReshapeAtDim(inputType.getRank(), dimsToDrop);

    SmallVector<int64_t> targetShape;
    for (int64_t k = 0; k < inputType.getRank(); ++k) {
      if (!llvm::is_contained(dimsToDrop, k)) {
        targetShape.push_back(inputType.getShape()[k]);
      }
    }

    auto collapsedType =
        RankedTensorType::get(targetShape, inputType.getElementType());

    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, collapsedType, value, reassociation);

    value = collapseOp.getResult();
    map = map.dropResults(dimsToDrop);
    return success();
  }

  FailureOr<Value> materializeBroadcastForOperand(PatternRewriter& rewriter,
                                                  linalg::GenericOp genericOp,
                                                  Value value, AffineMap map,
                                                  int64_t numDims) const {
    SmallVector<int64_t> addedDims = getBroadcastDimensions(map, numDims);
    int64_t targetRank = numDims - addedDims.size();

    if (failed(tryCollapseUnitDims(rewriter, genericOp.getLoc(), value, map,
                                   targetRank))) {
      return failure();
    }

    auto refOutput = genericOp.getDpsInitOperand(0)->get();
    auto refOutputType = cast<RankedTensorType>(refOutput.getType());

    auto emptyOp = tensor::EmptyOp::create(rewriter, genericOp.getLoc(),
                                           refOutputType.getShape(),
                                           refOutputType.getElementType());

    auto broadcastOp = linalg::BroadcastOp::create(
        rewriter, genericOp.getLoc(), value, emptyOp.getResult(), addedDims);

    return broadcastOp.getResults()[0];
  }
};

struct DropCfAssertInLinalg : public OpRewritePattern<linalg::GenericOp> {
 public:
  DropCfAssertInLinalg(MLIRContext* context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    bool madeChanges = false;
    auto* body = genericOp.getBody();
    for (auto& op : llvm::make_early_inc_range(body->getOperations())) {
      if (auto assertOp = dyn_cast<cf::AssertOp>(op)) {
        rewriter.eraseOp(assertOp);
        madeChanges = true;
      }
    }
    if (madeChanges) return success();
    return failure();
  }
};

// Rewrites a linalg.conv_1d_ncw_fcw operation with dilation > 1 into an
// equivalent dilation=1 convolution This materializes the filter to insert
// `dilation-1` zeros between the filter entries See
// https://github.com/vdumoulin/conv_arithmetic/blob/af6f818b0bb396c26da79899554682a8a499101d/gif/dilation.gif
// for a visualization of dilation
struct UndilateConv1DNcwFcw
    : public OpRewritePattern<mlir::linalg::Conv1DNcwFcwOp> {
 public:
  UndilateConv1DNcwFcw(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::Conv1DNcwFcwOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::Conv1DNcwFcwOp convOp,
                                PatternRewriter& rewriter) const override {
    auto dilations = convOp.getDilations();
    if (!dilations)
      return rewriter.notifyMatchFailure(convOp, "no dilations attribute");
    int64_t d = *dilations.getValues<int64_t>().begin();

    if (d == 1) return rewriter.notifyMatchFailure(convOp, "already undilated");

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    auto filterTy = cast<RankedTensorType>(filter.getType());
    if (!filterTy.hasStaticShape())
      return rewriter.notifyMatchFailure(convOp, "dynamic filter shape");

    // filter is (F, C, KW); dilation acts on KW. New KW' = d*(KW-1) + 1.
    ArrayRef<int64_t> shape = filterTy.getShape();
    int64_t f = shape[0], c = shape[1], kw = shape[2];
    int64_t newKw = d * (kw - 1) + 1;
    Type eltTy = filterTy.getElementType();
    auto newFilterTy = filterTy.clone({f, c, newKw});

    TypedAttr constAttr;
    if (!matchPattern(filter, m_Constant(&constAttr)))
      return rewriter.notifyMatchFailure(convOp, "filter must be a constant");

    auto denseAttr = dyn_cast<DenseElementsAttr>(constAttr);

    // handle denseResource
    if (auto denseResourceAttr =
            dyn_cast<DenseResourceElementsAttr>(constAttr)) {
      const auto data = denseResourceAttr.getData();
      // Limit size to avoid huge constants in IR.
      if (data.size() > 1024 * 1024)
        return rewriter.notifyMatchFailure(convOp,
                                           "filter too large to undilate");
      denseAttr = DenseElementsAttr::getFromRawBuffer(
          denseResourceAttr.getType(), data);
    }
    if (!denseAttr)
      return rewriter.notifyMatchFailure(
          convOp, "filter constant must be a dense elements attribute");

    SmallVector<Attribute> values(f * c * newKw, rewriter.getZeroAttr(eltTy));
    auto inputValues = denseAttr.getValues<Attribute>();
    for (int64_t fi = 0; fi < f; ++fi) {
      for (int64_t ci = 0; ci < c; ++ci) {
        for (int64_t ki = 0; ki < kw; ++ki) {
          int64_t src = fi * (c * kw) + ci * kw + ki;
          int64_t dst = fi * (c * newKw) + ci * newKw + ki * d;
          values[dst] = inputValues[src];
        }
      }
    }
    Value newFilter = arith::ConstantOp::create(
        rewriter, convOp.getLoc(), DenseElementsAttr::get(newFilterTy, values));

    rewriter.replaceOpWithNewOp<linalg::Conv1DNcwFcwOp>(
        convOp, convOp.getResultTypes(), ValueRange{input, newFilter},
        ValueRange{convOp.getDpsInits()[0]}, convOp.getStrides(),
        rewriter.getI64TensorAttr({1}));
    return success();
  }
};

// Rewrites a linalg.conv_2d_nchw_fchw operation with dilation > 1 into an
// equivalent dilation=1 convolution This materializes the filter to insert
// `dilation-1` zeros between the filter entries See
// https://github.com/vdumoulin/conv_arithmetic/blob/af6f818b0bb396c26da79899554682a8a499101d/gif/dilation.gif
// for a visualization of dilation
struct UndilateConv2DNchwFchw
    : public OpRewritePattern<mlir::linalg::Conv2DNchwFchwOp> {
 public:
  UndilateConv2DNchwFchw(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::Conv2DNchwFchwOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter& rewriter) const override {
    auto dilations = convOp.getDilations();
    if (!dilations)
      return rewriter.notifyMatchFailure(convOp, "no dilations attribute");
    SmallVector<int64_t> dilationValues =
        llvm::to_vector(dilations.getValues<int64_t>());

    int64_t dh = dilationValues[0];
    int64_t dw =
        (dilationValues.size() > 1) ? dilationValues[1] : dilationValues[0];

    if (dh == 1 && dw == 1)
      return rewriter.notifyMatchFailure(convOp, "already undilated");

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    auto filterTy = cast<RankedTensorType>(filter.getType());
    if (!filterTy.hasStaticShape())
      return rewriter.notifyMatchFailure(convOp, "dynamic filter shape");

    // filter is (F, C, KH, KW); dilation acts on (KH, KW).
    // New KH' = dh*(KH-1) + 1 and KW' = dw*(KW-1) + 1.
    ArrayRef<int64_t> shape = filterTy.getShape();
    int64_t f = shape[0], c = shape[1], kh = shape[2], kw = shape[3];
    int64_t newKh = dh * (kh - 1) + 1;
    int64_t newKw = dw * (kw - 1) + 1;
    Type eltTy = filterTy.getElementType();
    auto newFilterTy = filterTy.clone({f, c, newKh, newKw});

    TypedAttr constAttr;
    if (!matchPattern(filter, m_Constant(&constAttr)))
      return rewriter.notifyMatchFailure(convOp, "filter must be a constant");

    auto denseAttr = dyn_cast<DenseElementsAttr>(constAttr);

    // handle denseResource
    if (auto denseResourceAttr =
            dyn_cast<DenseResourceElementsAttr>(constAttr)) {
      const auto data = denseResourceAttr.getData();
      // Limit size to avoid huge constants in IR.
      if (data.size() > 1024 * 1024)
        return rewriter.notifyMatchFailure(convOp,
                                           "filter too large to undilate");
      denseAttr = DenseElementsAttr::getFromRawBuffer(
          denseResourceAttr.getType(), data);
    }
    if (!denseAttr)
      return rewriter.notifyMatchFailure(
          convOp, "filter constant must be a dense elements attribute");

    SmallVector<Attribute> values(f * c * newKh * newKw,
                                  rewriter.getZeroAttr(eltTy));
    auto inputValues = denseAttr.getValues<Attribute>();
    for (int64_t fi = 0; fi < f; ++fi) {
      for (int64_t ci = 0; ci < c; ++ci) {
        for (int64_t khi = 0; khi < kh; ++khi) {
          for (int64_t kwi = 0; kwi < kw; ++kwi) {
            int64_t src = ((fi * c + ci) * kh + khi) * kw + kwi;
            int64_t dst = ((fi * c + ci) * newKh + khi * dh) * newKw + kwi * dw;
            values[dst] = inputValues[src];
          }
        }
      }
    }
    Value newFilter = arith::ConstantOp::create(
        rewriter, convOp.getLoc(), DenseElementsAttr::get(newFilterTy, values));

    rewriter.replaceOpWithNewOp<linalg::Conv2DNchwFchwOp>(
        convOp, convOp.getResultTypes(), ValueRange{input, newFilter},
        ValueRange{convOp.getDpsInits()[0]}, convOp.getStrides(),
        rewriter.getI64TensorAttr({1, 1}));
    return success();
  }
};

struct LinalgCanonicalizations
    : public impl::LinalgCanonicalizationsBase<LinalgCanonicalizations> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<
        BroadcastToExpandShape, DropCfAssertInLinalg, FoldBroadcastExtractSlice,
        FoldConstantBroadcast, FoldConstantFill, FoldConstantLinalgTranspose,
        LinalgGenericToElementwise, LinalgMapToElementwise,
        MaterializeBroadcasts, RewriteAvgPoolAsConv1D, RewriteAvgPoolAsConv2D,
        RewriteTransposedMatvec, RewriteTransposedVecmat, UndilateConv1DNcwFcw,
        UndilateConv2DNchwFchw>(context);

    mlir::linalg::populateDecomposeProjectedPermutationPatterns(patterns);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
