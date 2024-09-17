#include "lib/Transforms/LinalgCanonicalizations/LinalgCanonicalizations.h"

#include <cstdint>
#include <utility>

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
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
  FoldConstantLinalgTranspose(MLIRContext *context)
      : OpRewritePattern<mlir::linalg::TransposeOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
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
      auto transposedConstantOp = rewriter.create<arith::ConstantOp>(
          transposeOp.getLoc(), transposedDenseElementsAttr);
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
    SmallVector<int64_t> transposedIndices;
    transposedIndices.reserve(inputShape.size());
    for (int64_t dim = 0; dim < inputShape.size(); ++dim) {
      int64_t dimSize = inputShape[permutation[dim]];
      remainingTensorSize /= dimSize;
      int64_t indexAtDim = remainder / remainingTensorSize;
      remainder %= remainingTensorSize;
      transposedIndices.push_back(indexAtDim);
    }

    // Convert to the original indices by applying the permutation.
    SmallVector<int64_t> originalIndices(permutation.size(), 0);
    for (int64_t dim = 0; dim < inputShape.size(); ++dim) {
      originalIndices[permutation[dim]] = transposedIndices[dim];
    }

    // Then compute the index in the original tensor.
    int64_t originalIndex = 0;
    int64_t remainingTransposedTensorSize = totalSize;
    for (int64_t dim = 0; dim < inputShape.size(); ++dim) {
      remainingTransposedTensorSize /= inputShape[dim];
      originalIndex += originalIndices[dim] * remainingTransposedTensorSize;
    }
    return originalIndex;
  }
};

struct LinalgCanonicalizations
    : public impl::LinalgCanonicalizationsBase<LinalgCanonicalizations> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldConstantLinalgTranspose>(context);

    // Run pattern matching and conversion
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
