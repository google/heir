#include "lib/Conversion/LinalgToTensorExt/LinalgToTensorExt.h"

#include <cstdint>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/APInt.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "linalg-to-arith"

namespace mlir {
namespace heir {
namespace linalg {

#define GEN_PASS_DEF_LINALGTOTENSOREXT
#include "lib/Conversion/LinalgToTensorExt/LinalgToTensorExt.h.inc"

bool isPowerOfTwo(int64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

int calculateIndexHelper(bool isLeftOperandSecret, int dim0, int dim1, int i,
                         int j) {
  // Note that we are producing the transpose of the diagonalized matrix.
  if (isLeftOperandSecret) {
    return ((i + j) % dim0) * dim1 + (j % dim1);
  } else {  // right operand is secret
    return (i % dim0) * dim0 + ((i + j) % dim1);
  }
}

template <typename T>
Value diagonalizeMatrix(ImplicitLocOpBuilder builder,
                        DenseElementsAttr denseAttr, bool isLeftOperandSecret) {
  // Algorithm for diagonalizing the matrix:
  // There are two loops, an outer loop and an inner loop.
  // The outer loop is the for loop that goes from 0 to number of rows in the
  // diagonalized, transposed matrix (transposedDimensions[0]).
  // The inner loop is the for loop that goes from 0 to number of columns in
  // the diagonalized, transposed matrix (transposedDimensions[1]).
  // At each iteration of the inner loop, we extract the correct diagonal
  // element from the matrix. Let's take an example:
  //
  // If the matrix is (for vector-matrix multiplication):
  // 1 2 3
  // 4 5 6
  // 7 8 9
  //
  // The the elements are laid out as follows in a list:
  // 1 2 3 4 5 6 7 8 9
  //
  // The diagonalized matrix as follows:
  // 1 5 9
  // 4 8 3
  // 7 2 6
  //
  // Then the diagonalized elements need to be laid out as follows:
  // 1 5 9 4 8 3 7 2 6
  //
  // Below is the algorithm for vector-matrix multiplication, where dim0 and
  // dim1 are the number of rows and columns in the matrix before
  // diagonalization. Note that calculating the diagonal index to extract
  // differs between vector-matrix multiplication and matrix-vector
  // multiplication (which is done via the helper function
  // calculateIndexHelper):
  //
  // for i = 0 to transposedDimensions[0]:
  //   for j = 0 to transposedDimensions[1]:
  //     row_index = (i + j) % dim0
  //     column_index = j % dim1
  //     index = row_index * dim1 + column_index
  //     diagonal_element = matrix[index]
  //     diagonal_elements.push_back(diagonal_element)
  //
  // Finally, we create a new constant op with the diagonalized elements.

  auto type = denseAttr.getElementType();

  auto dims = denseAttr.getType().getShape();
  auto dim0 = dims[0];
  auto dim1 = dims[1];
  SmallVector<int64_t> transposedDimensions({dim1, dim0});

  SmallVector<T> diagonalElements;
  diagonalElements.reserve(denseAttr.getNumElements());
  for (int i = 0; i < transposedDimensions[0]; ++i) {
    for (int j = 0; j < transposedDimensions[1]; ++j) {
      int index = calculateIndexHelper(isLeftOperandSecret, dim0, dim1, i, j);
      auto value = denseAttr.getValues<T>()[index];
      diagonalElements.push_back(value);
    }
  }
  auto diagonalizedType = RankedTensorType::get(transposedDimensions, type);
  auto diagonalizedDenseElementsAttr =
      DenseElementsAttr::get(diagonalizedType, diagonalElements);
  return builder.create<arith::ConstantOp>(diagonalizedDenseElementsAttr);
}

template <typename AddOp, typename MulOp>
Value multiplyDiagonalizedMatrixWithVector(ImplicitLocOpBuilder builder,
                                           Value diagonalizedMatrix,
                                           Value secretValues, Value bias,
                                           bool isLeftOperandSecret) {
  // The code below emits the following code for vector-matrix
  // multiplication (matrix-vector multiplication is similar):
  // %sum = bias
  // %rotated_vector = secretValues
  // for %i = 0 to transposedDim0 - 1:
  //   %extractedSlice = extract_slice %newMatrix[%i, 0] [1, transposedDim1] [1,
  //   1]
  //   %multiplied = %rotated_vector * %extractedSlice %sum = %sum + %multiplied
  //   %rotated_vector = rotate %rotated_vector, 1
  // %lastExtracted = extract_slice %newMatrix[transposedDim0-1, 0] [1,
  // transposedDim1] [1, 1]
  // %final_sum = %sum + %lastExtracted
  // At this point, we can rotate and sum if needed. (Squat packing is left as a
  // TODO until we resolve the shape mismatch issue.)
  // return %final_sum

  auto shape = cast<RankedTensorType>(diagonalizedMatrix.getType()).getShape();
  auto transposedDim0 = shape[0];
  auto transposedDim1 = shape[1];

  // Build a constant index 1.
  auto indexOne = builder.create<arith::ConstantIndexOp>(1);

  // Setup sizes and strides, which are parameters for the tensor_ext
  // ExtractSliceOp.
  SmallVector<OpFoldResult> sizes(2);
  if (isLeftOperandSecret) {
    sizes = {builder.getIndexAttr(1), builder.getIndexAttr(transposedDim1)};
  } else {
    sizes = {builder.getIndexAttr(transposedDim0), builder.getIndexAttr(1)};
  }
  SmallVector<OpFoldResult> strides(2, builder.getIndexAttr(1));

  // Setup parameters for the affine for loop.
  SmallVector<Value> iterArgs({bias, secretValues});
  int numLoops;
  if (isLeftOperandSecret) {
    numLoops = transposedDim0;
  } else {  // right operand is secret
    numLoops = transposedDim1;
  }

  // Build the affine for loop.
  auto forOp =
      builder.create<mlir::affine::AffineForOp>(0, numLoops - 1, 1, iterArgs);

  // Now, we are inside for loop.
  builder.setInsertionPointToStart(forOp.getBody());
  auto index = forOp.getInductionVar();
  auto sum = forOp.getRegionIterArgs()[0];
  auto rotatedVector = forOp.getRegionIterArgs()[1];

  // Setup the offsets for the ExtractSliceOp and build the ExtractSliceOp.
  SmallVector<OpFoldResult> offsets(2);
  if (isLeftOperandSecret) {
    offsets = {index, builder.getIndexAttr(0)};
  } else {
    offsets = {builder.getIndexAttr(0), index};
  }
  auto extracted = builder.create<tensor::ExtractSliceOp>(
      diagonalizedMatrix, offsets, sizes, strides);

  // Calculates:
  // 1) multiplied = rotatedVector * extracted. (This is a pairwise scalar
  // multiplication.)
  // 2) newSum = sum + multiplied. (This is a pairwise addition.)
  // 3) newRotatedVector = rotate(rotatedVector, 1).
  // 4) returns (newSum, newRotatedVector).

  auto multiplied = builder.create<MulOp>(rotatedVector, extracted);
  auto newSum = builder.create<AddOp>(sum, multiplied);
  auto newRotatedVector =
      builder.create<tensor_ext::RotateOp>(rotatedVector, indexOne);
  builder.create<affine::AffineYieldOp>(ValueRange({newSum, newRotatedVector}));

  // Now outside for loop.
  builder.setInsertionPointAfter(forOp);

  // Setup the offsets for the last ExtractSliceOp and build the
  // ExtractSliceOp.
  SmallVector<OpFoldResult> lastOffsets(2);
  if (isLeftOperandSecret) {
    lastOffsets = {builder.getIndexAttr(transposedDim1 - 1),
                   builder.getIndexAttr(0)};
  } else {
    lastOffsets = {builder.getIndexAttr(0),
                   builder.getIndexAttr(transposedDim1 - 1)};
  }
  auto lastExtracted = builder.create<tensor::ExtractSliceOp>(
      diagonalizedMatrix, lastOffsets, sizes, strides);

  // Calculates the final scalar multiplication and sum.
  auto lastMultiplied =
      builder.create<MulOp>(forOp.getResults()[1], lastExtracted);
  auto finalSum = builder.create<AddOp>(forOp.getResults()[0], lastMultiplied);
  return finalSum;
}

struct ConvertLinalgMatmul : public OpRewritePattern<mlir::linalg::MatmulOp> {
 private:
  DataFlowSolver *solver;

 public:
  ConvertLinalgMatmul(DataFlowSolver *solver, mlir::MLIRContext *context)
      : OpRewritePattern<mlir::linalg::MatmulOp>(context), solver(solver) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // Determine if the left or right operand is secret to determine which
    // matrix to diagonalize, or if both are secret or both are public, then
    // return failure.
    auto isSecret = [&](Value value) {
      auto *operandLookup = solver->lookupState<SecretnessLattice>(value);
      Secretness operandSecretness =
          operandLookup ? operandLookup->getValue() : Secretness();
      return (operandSecretness.isInitialized() &&
              operandSecretness.getSecretness());
    };

    bool isLeftOperandSecret = isSecret(op.getInputs()[0]);
    bool isRightOperandSecret = isSecret(op.getInputs()[1]);

    LLVM_DEBUG({
      llvm::dbgs() << "Left operand is secret: " << isLeftOperandSecret << "\n"
                   << "Right operand is secret: " << isRightOperandSecret
                   << "\n";
    });

    // Error out if both are secret or both are public
    if ((isLeftOperandSecret && isRightOperandSecret) ||
        (!isLeftOperandSecret && !isRightOperandSecret)) {
      return failure();
    }
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();

    // Assign if the left operand is secret
    Value secretValues = inputs[0];
    Value publicMatrix = inputs[1];
    if (isRightOperandSecret) {
      std::swap(secretValues, publicMatrix);
    }
    auto matrixTensorType = cast<RankedTensorType>(publicMatrix.getType());
    auto bias = outputs[0];

    auto dimensions = matrixTensorType.getShape();
    int64_t dim0 = dimensions[0];  // This is the number of rows in the matrix.
    int64_t dim1 = dimensions[1];  // This is the number of columns in the
    // matrix.

    // If one of these dimensions is not a power of two, then we can't do the
    // Halevi-Shoup or Squat Packing Matrix Multiplication conversion.
    if (!isPowerOfTwo(dim0) || !isPowerOfTwo(dim1)) {
      return failure();
    }

    // If the matrix is not a square matrix, then we are doing squat packing.
    // TODO: Implement squat packing.
    if (dim0 != dim1) {
      return failure();
    }

    // Diagonalize the matrix only if the matrix is a constant.
    auto constantValues =
        dyn_cast<arith::ConstantOp>(publicMatrix.getDefiningOp());
    if (!constantValues) {
      return failure();
    }

    DenseElementsAttr denseAttr =
        dyn_cast<DenseElementsAttr>(constantValues.getValueAttr());

    // If the constant values doesn't have a dense attribute, then we can't
    // diagonalize the matrix.
    if (!denseAttr) {
      return failure();
    }

    auto type = denseAttr.getElementType();
    if (!type.isIntOrFloat()) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value result;

    // First, modify the matrix to be a diagonal matrix. We'll simply create a
    // copy of the weight matrix diagonalized, and if the old weight matrix is
    // not used, then dead code elimination pass will remove it.

    // After that, we create code for multiplying the matrix with rotations of
    // the vector.
    if (type.isInteger()) {
      Value diagonalizedMatrix =
          diagonalizeMatrix<APInt>(b, denseAttr, isLeftOperandSecret);

      result =
          multiplyDiagonalizedMatrixWithVector<arith::AddIOp, arith::MulIOp>(
              b, diagonalizedMatrix, secretValues, bias, isLeftOperandSecret);
    } else {  // floating point
      Value diagonalizedMatrix =
          diagonalizeMatrix<APFloat>(b, denseAttr, isLeftOperandSecret);

      result =
          multiplyDiagonalizedMatrixWithVector<arith::AddFOp, arith::MulFOp>(
              b, diagonalizedMatrix, secretValues, bias, isLeftOperandSecret);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LinalgToTensorExt
    : public impl::LinalgToTensorExtBase<LinalgToTensorExt> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(module);

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);

    patterns.add<ConvertLinalgMatmul>(&solver, context);

    // Run pattern matching and conversion
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace linalg
}  // namespace heir
}  // namespace mlir
