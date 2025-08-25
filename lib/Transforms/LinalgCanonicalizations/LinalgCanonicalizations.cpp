#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"

#include <cstdint>
#include <utility>

#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
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
    auto dest = mapOp.getInit().getDefiningOp<tensor::EmptyOp>();
    if (!dest) return failure();

    // The mapper should have exactly two operations (the second is a yield).
    auto mapper = mapOp.getBody(0);
    if (mapper->getOperations().size() != 2) return failure();

    // The operation should be elementwise.
    Operation& op = mapper->getOperations().front();
    if (!op.hasTrait<mlir::OpTrait::Elementwise>()) return failure();

    auto elementwiseOp = rewriter.create(
        mapOp->getLoc(), op.getName().getIdentifier(), mapOp.getInputs(),
        TypeRange(dest.getType()), mapOp->getAttrs(), {}, {});
    rewriter.replaceOp(mapOp, elementwiseOp);
    if (dest.use_empty()) rewriter.eraseOp(dest);
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

struct LinalgCanonicalizations
    : public impl::LinalgCanonicalizationsBase<LinalgCanonicalizations> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldConstantLinalgTranspose, FoldConstantFill,
                 FoldConstantBroadcast, LinalgMapToElementwise,
                 BroadcastToExpandShape, RewriteTransposedVecmat,
                 RewriteTransposedMatvec>(context);

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
