#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"          // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
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

    auto defConstantOp =
        transposeOp.getInput().getDefiningOp<arith::ConstantOp>();
    if (!defConstantOp)
      return rewriter.notifyMatchFailure(transposeOp,
                                         "transpose input must be a constant");
    DenseElementsAttr denseAttr =
        dyn_cast<DenseElementsAttr>(defConstantOp.getValueAttr());
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
    auto outputTy = cast<RankedTensorType>(broadcastOp.getResultTypes()[0]);

    // Replace with the new broadcasted constant.
    // For now, only handle splats.
    auto splatInput = dyn_cast<SplatElementsAttr>(cast<Attribute>(value));
    if (!splatInput)
      return rewriter.notifyMatchFailure(
          broadcastOp, "broadcast input must be a splat attribute");

    auto broadcastedInput =
        SplatElementsAttr::get(outputTy, splatInput.getSplatValue<Attribute>());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(broadcastOp, outputTy,
                                                   broadcastedInput);
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
      auto* elementwiseOp = rewriter.create(
          genericOp->getLoc(), innerOp.getName().getIdentifier(), newInputs,
          newDestTypes, innerOp.getAttrs(), {}, {});
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

    auto kernelShape =
        SmallVector<int64_t>{filterTy.getDimSize(0), filterTy.getDimSize(1)};
    auto kernelTy =
        RankedTensorType::get(kernelShape, filterTy.getElementType());
    TypedAttr kernelVals = rewriter.getOneAttr(kernelTy);

    // If there is a constant division following the sum pool, update the
    // kernel of ones to be 1 / divValue. This is a common enough pattern since
    // it represents an average pool.
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
            kernelVals = SplatElementsAttr::get(kernelTy, one / divValue);
            avgPoolOutput = divOp.getResult();
          }
        }
      }
    }
    auto kernel =
        arith::ConstantOp::create(rewriter, poolOp.getLoc(), kernelVals);

    // Rewrite the 2D avg pool output shape is N x C x H' x W'. Apply the kernel
    // to each channel separately, and insert into the output.
    auto outputVal = poolOp.getOutputs()[0];
    //  The filter must ensure each output channel i is only the sum of values
    //  from input channel i. So the filter uses an identity matrix when f ==
    //  c and 0 otherwise.
    RankedTensorType twoDOutputType =
        RankedTensorType::get({outputTy.getDimSize(2), outputTy.getDimSize(3)},
                              outputTy.getElementType());
    RankedTensorType twoDInputType =
        RankedTensorType::get({inputTy.getDimSize(2), inputTy.getDimSize(3)},
                              inputTy.getElementType());
    Value convOutput = tensor::EmptyOp::create(rewriter, poolOp.getLoc(),
                                               twoDOutputType.getShape(),
                                               twoDOutputType.getElementType());
    for (int n = 0; n < inputTy.getDimSize(0); ++n) {
      for (int c = 0; c < inputTy.getDimSize(1); ++c) {
        // Compute the 2-D constant convolution.
        SmallVector<OpFoldResult> offsets = {
            rewriter.getIndexAttr(n), rewriter.getIndexAttr(c),
            rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult> inputSizes = {
            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(inputTy.getDimSize(2)),
            rewriter.getIndexAttr(inputTy.getDimSize(3))};
        SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
        auto extractInputOp = tensor::ExtractSliceOp::create(
            rewriter, poolOp.getLoc(), twoDInputType, poolOp.getInputs()[0],
            offsets, inputSizes, strides);

        auto convOp = linalg::Conv2DOp::create(
            rewriter, poolOp.getLoc(), twoDOutputType,
            ValueRange{extractInputOp, kernel}, ValueRange{convOutput});
        // Insert into the outputVal.
        SmallVector<OpFoldResult> outputSizes = {
            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(outputTy.getDimSize(2)),
            rewriter.getIndexAttr(outputTy.getDimSize(3))};
        outputVal = tensor::InsertSliceOp::create(
            rewriter, poolOp.getLoc(), convOp.getResult(0), outputVal, offsets,
            outputSizes, strides);
      }
    }

    if (avgPoolOutput) {
      rewriter.replaceAllUsesWith(avgPoolOutput, outputVal);
    } else {
      rewriter.replaceOp(poolOp, outputVal);
    }
    return success();
  }
};

// Lower linalg.conv_2d_nchw_fchw to a loop of linalg.conv_2d operations.
struct LowerConv2DNchwFchw
    : public OpRewritePattern<mlir::linalg::Conv2DNchwFchwOp> {
 public:
  LowerConv2DNchwFchw(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::Conv2DNchwFchwOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter& rewriter) const override {
    Location loc = convOp.getLoc();
    Value image = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    // The output tensor is the initial value for the result.
    Value output = convOp.getOutputs()[0];

    auto imageType = cast<RankedTensorType>(image.getType());
    auto filterType = cast<RankedTensorType>(filter.getType());
    auto outputType = cast<RankedTensorType>(output.getType());

    auto imageShape = imageType.getShape();
    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();

    int64_t n = imageShape[0];
    int64_t c = imageShape[1];
    int64_t f = filterShape[0];

    RankedTensorType twoDType = RankedTensorType::get(
        {imageShape[2], imageShape[3]}, imageType.getElementType());
    RankedTensorType outputTwoDType = RankedTensorType::get(
        {outputShape[2], outputShape[3]}, outputType.getElementType());
    RankedTensorType filterTwoDType = RankedTensorType::get(
        {filterShape[2], filterShape[3]}, filterType.getElementType());

    // Create loops over the batch and filter dimensions.
    // The loops carry the output tensor being updated.
    auto nLoop = affine::AffineForOp::create(
        rewriter, loc, 0, n, 1, ValueRange{output},
        [&](OpBuilder& b, Location loc, Value nIv, ValueRange iterArgs) {
          Value iterOutput = iterArgs[0];
          auto fLoop = affine::AffineForOp::create(
              b, loc, 0, f, 1, ValueRange{iterOutput},
              [&](OpBuilder& b, Location loc, Value fIv, ValueRange iterArgs) {
                // Slice the output to get an init tensor for the convolution
                // results on each input channel.
                Value innerIterOutput = iterArgs[0];
                SmallVector<OpFoldResult> outputOffsets = {
                    nIv, fIv, b.getIndexAttr(0), b.getIndexAttr(0)};
                SmallVector<OpFoldResult> outputSizes = {
                    b.getIndexAttr(1), b.getIndexAttr(1),
                    b.getIndexAttr(outputShape[2]),
                    b.getIndexAttr(outputShape[3])};
                SmallVector<OpFoldResult> outputStrides(4, b.getIndexAttr(1));
                Value outputSlice = tensor::ExtractSliceOp::create(
                    b, loc, outputTwoDType, innerIterOutput, outputOffsets,
                    outputSizes, outputStrides);

                // Accumulate over each of the input channels.
                auto cLoop = affine::AffineForOp::create(
                    b, loc, 0, c, 1, ValueRange{outputSlice},
                    [&](OpBuilder& b, Location loc, Value cIv,
                        ValueRange iterArgs) {
                      // Slice the image for the current batch.
                      SmallVector<OpFoldResult> imageOffsets = {
                          nIv, cIv, b.getIndexAttr(0), b.getIndexAttr(0)};
                      SmallVector<OpFoldResult> imageSizes = {
                          b.getIndexAttr(1), b.getIndexAttr(1),
                          b.getIndexAttr(imageShape[2]),
                          b.getIndexAttr(imageShape[3])};
                      SmallVector<OpFoldResult> imageStrides(4,
                                                             b.getIndexAttr(1));
                      Value imageSlice = tensor::ExtractSliceOp::create(
                          b, loc, twoDType, image, imageOffsets, imageSizes,
                          imageStrides);

                      // Slice the filter for the current filter.
                      SmallVector<OpFoldResult> filterOffsets = {
                          fIv, cIv, b.getIndexAttr(0), b.getIndexAttr(0)};
                      SmallVector<OpFoldResult> filterSizes = {
                          b.getIndexAttr(1), b.getIndexAttr(1),
                          b.getIndexAttr(filterShape[2]),
                          b.getIndexAttr(filterShape[3])};
                      SmallVector<OpFoldResult> filterStrides(
                          4, b.getIndexAttr(1));
                      Value filterSlice = tensor::ExtractSliceOp::create(
                          b, loc, filterTwoDType, filter, filterOffsets,
                          filterSizes, filterStrides);

                      // Create a 1x1 conv op for the slices.
                      Value convOutput = tensor::EmptyOp::create(
                          b, loc, outputTwoDType.getShape(),
                          twoDType.getElementType());
                      auto conv = linalg::Conv2DOp::create(
                          b, loc, outputTwoDType,
                          ValueRange{imageSlice, filterSlice},
                          ValueRange{convOutput});

                      Value accumulatedOutput = arith::AddFOp::create(
                          b, loc, conv.getResult(0), iterArgs[0]);
                      affine::AffineYieldOp::create(b, loc, accumulatedOutput);
                    });

                // Insert the result of the conv back into the output tensor.
                Value newOutput = tensor::InsertSliceOp::create(
                    b, loc, cLoop.getResult(0), innerIterOutput, outputOffsets,
                    outputSizes, outputStrides);
                affine::AffineYieldOp::create(b, loc, newOutput);
              });
          affine::AffineYieldOp::create(b, loc, fLoop.getResults());
        });

    rewriter.replaceOp(convOp, nLoop.getResults());
    return success();
  }
};

struct LinalgCanonicalizations
    : public impl::LinalgCanonicalizationsBase<LinalgCanonicalizations> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldConstantLinalgTranspose, FoldConstantFill,
                 FoldConstantBroadcast, LinalgMapToElementwise,
                 LinalgGenericToElementwise, BroadcastToExpandShape,
                 RewriteTransposedVecmat, RewriteTransposedMatvec,
                 RewriteAvgPoolAsConv2D, LowerConv2DNchwFchw>(context);

    // Run pattern matching and conversion
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
