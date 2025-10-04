#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
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
    if (!defConstantOp) return failure();
    DenseElementsAttr denseAttr =
        dyn_cast<DenseElementsAttr>(defConstantOp.getValueAttr());
    if (!denseAttr) return failure();

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
    return failure();
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
    if (isa<Value>(value)) return failure();
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
    if (isa<Value>(value)) return failure();
    auto outputTy = cast<RankedTensorType>(broadcastOp.getResultTypes()[0]);

    // Replace with the new broadcasted constant.
    // For now, only handle splats.
    auto splatInput = dyn_cast<SplatElementsAttr>(cast<Attribute>(value));
    if (!splatInput) return failure();

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
    if (mapper->getOperations().size() != 2) return failure();

    // The operation should be elementwise.
    Operation& op = mapper->getOperations().front();
    if (!op.hasTrait<mlir::OpTrait::Elementwise>()) return failure();

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
      if (!op.hasTrait<mlir::OpTrait::Elementwise>()) {
        return rewriter.notifyMatchFailure(
            genericOp,
            "genericOp with non-elementwise operation not supported");
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

      auto resultTensorType =
          cast<RankedTensorType>(genericOp.getResult(0).getType());
      for (auto& operand : innerOp.getOpOperands()) {
        if (isa<BlockArgument>(operand.get())) {
          auto blockArgument = cast<BlockArgument>(operand.get());
          assert(blockArgument.getArgNumber() < genericOp.getInputs().size() &&
                 "expecting that the elementwise operation does not use the "
                 "output's initial values");
          // Otherwise, map the block argument to the operand of the generic op.
          bvm.map(operand.get(),
                  genericOp.getOperand(blockArgument.getArgNumber()));
        } else if (operand.get().getParentBlock() != genericOp.getBody()) {
          // If this isn't a block argument and wasn't defined in the generic,
          // then it must be a constant from outside the generic. The type of
          // the splat constant must match the result of the generic op since
          // all of the operations are elementwise.
          Value splatTensor = tensor::SplatOp::create(
              rewriter, genericOp.getLoc(), resultTensorType, operand.get());
          bvm.map(operand.get(), splatTensor);
        } else {
          // This was defined within the generic op so it must be a result of
          // some other innerOp and its value must have a mapping in bvm.
          assert(bvm.lookup(operand.get()) &&
                 "expected a mapping for this value");
        }
      }
      auto newInputs = llvm::to_vector(llvm::map_range(
          innerOp.getOperands(), [&](Value v) { return bvm.lookup(v); }));
      auto newDestTypes = llvm::to_vector(llvm::map_range(
          innerOp.getResultTypes(),
          [&](Type t) -> Type { return resultTensorType.clone(t); }));
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
    if (res != SliceVerificationResult::Success) return failure();

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
    if (!transposeOp) return failure();

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
    if (!transposeOp) return failure();

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
    // The kernel for the convolution will have shape M x N x H x W where M is
    // the number of channels of the output, and N is the number of input
    // channels.
    auto kernelShape =
        SmallVector<int64_t>{outputTy.getDimSize(1), inputTy.getDimSize(1),
                             filterTy.getDimSize(0), filterTy.getDimSize(1)};
    auto kernelTy =
        RankedTensorType::get(kernelShape, filterTy.getElementType());
    //  The filter must ensure each output channel i is only the sum of values
    //  from input channel i. So the filter uses an identity matrix when f ==
    //  c and 0 otherwise.
    SmallVector<APFloat> filterValues;
    filterValues.reserve(kernelTy.getNumElements());
    for (int f = 0; f < kernelShape[0]; ++f) {
      for (int c = 0; c < kernelShape[1]; ++c) {
        // Insert h * w 1s or 0s into the filter values.
        auto numKernelElements = kernelShape[2] * kernelShape[3];
        if (f == c) {
          filterValues.insert(filterValues.end(), numKernelElements,
                              APFloat(1.0f));
        } else {
          filterValues.insert(filterValues.end(), numKernelElements,
                              APFloat(0.0f));
        }
      }
    }
    auto constantKernel = arith::ConstantOp::create(
        rewriter, poolOp->getLoc(),
        mlir::DenseElementsAttr::get(kernelTy, filterValues));
    rewriter.replaceOpWithNewOp<linalg::Conv2DNchwFchwOp>(
        poolOp, poolOp.getResultTypes()[0],
        ValueRange{poolOp.getInputs()[0], constantKernel},
        poolOp.getDpsInits()[0], poolOp.getStridesAttr(),
        poolOp.getDilationsAttr());
    return success();
  }
};

struct FoldDivsAfterConstantConv
    : public OpRewritePattern<mlir::arith::DivFOp> {
 public:
  FoldDivsAfterConstantConv(MLIRContext* context)
      : OpRewritePattern<mlir::arith::DivFOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::arith::DivFOp arithOp,
                                PatternRewriter& rewriter) const override {
    // Check if one of the operands came from a convolution.
    int64_t convResultIndex;
    linalg::Conv2DNchwFchwOp convOp;
    for (OpOperand& operand : arithOp->getOpOperands()) {
      if (auto conv = operand.get().getDefiningOp<linalg::Conv2DNchwFchwOp>()) {
        convOp = conv;
        convResultIndex = operand.getOperandNumber();
        break;
      }
    }
    if (!convOp)
      return rewriter.notifyMatchFailure(
          arithOp, "DivOp not after constant convolution not supported");

    // Check if the other operand to the arith::div is a constant.
    auto constant = getAsOpFoldResult(arithOp.getOperand(1 - convResultIndex));
    if (!isa<Attribute>(constant))
      return rewriter.notifyMatchFailure(arithOp,
                                         "division operand must be a constant");
    auto constantAttr = cast<DenseElementsAttr>(cast<Attribute>(constant));
    // If the division is not with a splat elements, it may not be clear how
    // to fold this into the convolution kernel.
    if (!isa<SplatElementsAttr>(constantAttr)) {
      return rewriter.notifyMatchFailure(
          arithOp, "constant division after convolution must be a splat");
    }
    auto divValue = constantAttr.getSplatValue<APFloat>();

    // Ensure that the convolution kernel is a constant.
    auto kernel = getAsOpFoldResult(convOp.getInputs()[1]);
    if (!isa<Attribute>(kernel)) {
      return rewriter.notifyMatchFailure(
          arithOp, "convolution kernel must be a constant");
    }
    auto kernelAttr = cast<DenseElementsAttr>(cast<Attribute>(kernel));

    // Apply the constant division to each element of the convolution kernel.
    SmallVector<APFloat> filterValues;
    filterValues.reserve(kernelAttr.getNumElements());
    for (auto value : kernelAttr.getValues<APFloat>()) {
      filterValues.push_back(value / divValue);
    }

    auto constantKernel = arith::ConstantOp::create(
        rewriter, arithOp->getLoc(),
        mlir::DenseElementsAttr::get(kernelAttr.getType(), filterValues));
    rewriter.replaceOpWithNewOp<linalg::Conv2DNchwFchwOp>(
        arithOp, convOp.getResultTypes()[0],
        ValueRange{convOp.getInputs()[0], constantKernel},
        convOp.getDpsInits()[0], convOp.getStridesAttr(),
        convOp.getDilationsAttr());
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
                 RewriteAvgPoolAsConv2D, FoldDivsAfterConstantConv>(context);

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
