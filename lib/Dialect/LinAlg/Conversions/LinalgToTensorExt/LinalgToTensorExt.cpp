#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h"

#include <cstdint>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/APInt.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "linalg-to-tensor-ext"

namespace mlir {
namespace heir {
namespace linalg {

#define GEN_PASS_DEF_LINALGTOTENSOREXT
#include "lib/Dialect/LinAlg/Conversions/LinalgToTensorExt/LinalgToTensorExt.h.inc"

int calculateIndexHelper(bool isLeftOperandSecret, int dim0, int dim1, int i,
                         int j) {
  // Note that we are producing the transpose of the diagonalized matrix.
  if (isLeftOperandSecret) {
    return ((i + j) % dim0) * dim1 + (j % dim1);
  } else {  // right operand is secret
    return (i % dim0) * dim1 + ((i + j) % dim1);
  }
}

template <typename T>
Value diagonalizeMatrix(ImplicitLocOpBuilder builder,
                        DenseElementsAttr denseAttr, bool isLeftOperandSecret,
                        int maxTilingSize) {
  // Algorithm for diagonalizing the matrix into a square matrix of size
  // maxTilingSize x maxTilingSize.
  // There are two loops, an outer loop and an inner loop.
  // The outer loop is the for loop that goes from 0 to number of rows in the
  // diagonalized, transposed matrix (maxTilingSize).
  // The inner loop is the for loop that goes from 0 to number of columns in
  // the diagonalized, transposed matrix (maxTilingSize).
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
  // for i = 0 to maxTilingSize:
  //   for j = 0 to maxTilingSize:
  //     row_index = (i + j) % dim0
  //     column_index = j % dim1
  //     index = row_index * dim1 + column_index
  //     diagonal_element = matrix[index]
  //     diagonal_elements.push_back(diagonal_element)
  //
  // Finally, we create a new constant op with the diagonalized elements.

  auto type = denseAttr.getElementType();

  auto dims = denseAttr.getType().getShape();
  auto denseAttrValues = denseAttr.getValues<T>();
  auto dim0 = dims[0];
  auto dim1 = dims[1];
  SmallVector<int64_t> transposedDimensions({maxTilingSize, maxTilingSize});

  SmallVector<T> diagonalElements;
  diagonalElements.reserve(denseAttr.getNumElements());
  for (int i = 0; i < maxTilingSize; ++i) {
    for (int j = 0; j < maxTilingSize; ++j) {
      int index = calculateIndexHelper(isLeftOperandSecret, dim0, dim1, i, j);
      LLVM_DEBUG({
        llvm::dbgs() << "i: " << i << ", j: " << j << ", index: " << index
                     << ", dim0: " << dim0 << ", dim1: " << dim1 << "\n";
      });
      auto value = denseAttrValues[index];
      diagonalElements.push_back(value);
    }
  }
  auto diagonalizedType = RankedTensorType::get(transposedDimensions, type);
  auto diagonalizedDenseElementsAttr =
      DenseElementsAttr::get(diagonalizedType, diagonalElements);
  return builder.create<arith::ConstantOp>(diagonalizedDenseElementsAttr);
}

template <typename T>
Value duplicateBias(ImplicitLocOpBuilder builder, DenseElementsAttr biasAttr,
                    bool isLeftOperandSecret, int maxTilingSize) {
  auto type = biasAttr.getElementType();

  int numElements = biasAttr.getNumElements();

  SmallVector<int64_t> duplicatedDimensions({1, maxTilingSize});
  if (!isLeftOperandSecret) {
    duplicatedDimensions = {maxTilingSize, 1};
  }

  SmallVector<T> newBiasElements;
  newBiasElements.reserve(maxTilingSize);
  for (int i = 0; i < maxTilingSize; ++i) {
    newBiasElements.push_back(biasAttr.getValues<T>()[i % numElements]);
  }
  auto duplicatedBiasType = RankedTensorType::get(duplicatedDimensions, type);
  auto duplicatedBiasDenseElementsAttr =
      DenseElementsAttr::get(duplicatedBiasType, newBiasElements);
  return builder.create<arith::ConstantOp>(duplicatedBiasDenseElementsAttr);
}

template <typename AddOp, typename MulOp>
Value multiplyDiagonalizedMatrixWithVector(
    ImplicitLocOpBuilder builder, Value diagonalizedMatrix,
    ArrayRef<int64_t> originalMatrixDimensions, Value secretValues, Value bias,
    bool isLeftOperandSecret, int maxTilingSize) {
  // The code below emits the following code for vector-matrix
  // multiplication (matrix-vector multiplication is similar):
  // %sum = bias
  // %rotated_vector = secretValues
  // for %i = 0 to originalMatrixDimensions[1] - 1:
  //   %extractedSlice = extract_slice %newMatrix[%i, 0]
  //      [1, originalMatrixDimensions[0]] [1, 1]
  //   %multiplied = %rotated_vector * %extractedSlice
  //   %sum = %sum + %multiplied
  //   %rotated_vector = rotate %rotated_vector, 1
  // %lastExtracted = extract_slice %newMatrix[maxTilingSize-1, 0] [1,
  //      originalMatrixDimensions[0]] [1, 1]
  // %final_sum = %sum + %lastExtracted
  // At this point, we can rotate and sum if needed.
  // return %final_sum

  // Build a constant index 1.
  auto indexOne = builder.create<arith::ConstantIndexOp>(1);

  // Setup sizes and strides, which are parameters for the tensor_ext
  // ExtractSliceOp.
  SmallVector<OpFoldResult> sizes(2);
  if (isLeftOperandSecret) {
    sizes = {builder.getIndexAttr(1), builder.getIndexAttr(maxTilingSize)};
  } else {
    sizes = {builder.getIndexAttr(maxTilingSize), builder.getIndexAttr(1)};
  }
  SmallVector<OpFoldResult> strides(2, builder.getIndexAttr(1));

  // Setup the offsets for the last ExtractSliceOp and build the
  // ExtractSliceOp.
  SmallVector<OpFoldResult> firstOffsets(2, builder.getIndexAttr(0));
  auto firstExtracted = builder.create<tensor::ExtractSliceOp>(
      diagonalizedMatrix, firstOffsets, sizes, strides);

  // Calculates the first scalar multiplication and sum.
  auto firstMultiplied = builder.create<MulOp>(secretValues, firstExtracted);
  auto firstSumWithoutRotateAndSum =
      builder.create<AddOp>(bias, firstMultiplied);

  // Build the affine for loop.
  // Setup parameters for the affine for loop.
  int numLoops = originalMatrixDimensions[0];
  if (numLoops > originalMatrixDimensions[1]) {
    numLoops = originalMatrixDimensions[1];
  }

  SmallVector<Value> iterArgs({firstSumWithoutRotateAndSum, secretValues});
  auto forOp =
      builder.create<mlir::affine::AffineForOp>(1, numLoops, 1, iterArgs);

  // Now, we are inside for loop.
  builder.setInsertionPointToStart(forOp.getBody());
  auto index = forOp.getInductionVar();
  auto sum = forOp.getRegionIterArgs()[0];

  // Rotate first
  auto rotatedVector = builder.create<tensor_ext::RotateOp>(
      forOp.getRegionIterArgs()[1], indexOne);

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
  builder.create<affine::AffineYieldOp>(ValueRange({newSum, rotatedVector}));

  // Now outside for loop.
  builder.setInsertionPointAfter(forOp);

  int numRotationsAndSums;
  if (isLeftOperandSecret) {
    numRotationsAndSums = llvm::APInt(32, originalMatrixDimensions[0] /
                                              originalMatrixDimensions[1])
                              .exactLogBase2();
  } else {
    numRotationsAndSums = llvm::APInt(32, originalMatrixDimensions[1] /
                                              originalMatrixDimensions[0])
                              .exactLogBase2();
  }

  // Rotate and sum if needed
  Value sumInProgress = forOp.getResults()[0];
  int rotationValue = maxTilingSize;
  for (int i = 0; i < numRotationsAndSums; ++i) {
    rotationValue /= 2;
    auto rotatedTensor = builder.create<tensor_ext::RotateOp>(
        sumInProgress,
        builder.create<arith::ConstantOp>(builder.getIndexAttr(rotationValue)));
    sumInProgress = builder.create<AddOp>(sumInProgress, rotatedTensor);
  }

  return sumInProgress;
}

class ReplicatedTensorTypeConverter : public TypeConverter {
 private:
  int maxTilingSize;

 public:
  ReplicatedTensorTypeConverter(int maxTilingSize)
      : maxTilingSize(maxTilingSize) {
    addConversion([](Type type) { return type; });

    addConversion([this](RankedTensorType type) -> Type {
      if (type.getShape().size() == 1) {
        return type;
      }
      // Assuming 2-d operations only
      if (type.getShape()[0] == 1) {
        return RankedTensorType::get({1, this->maxTilingSize},
                                     type.getElementType());
      } else if (type.getShape()[1] == 1) {
        return RankedTensorType::get({this->maxTilingSize, 1},
                                     type.getElementType());
      } else {
        return RankedTensorType::get({this->maxTilingSize, this->maxTilingSize},
                                     type.getElementType());
      }
    });

    // Convert secret tensors to secret tensors of the right size.
    addConversion([this](secret::SecretType type) -> Type {
      return secret::SecretType::get(this->convertType(type.getValueType()));
    });
  }
};

// Returns true if the generic contains a matmul op that can be rewritten with
// squat packing.
bool isSquatPackableMatmul(secret::GenericOp genericOp,
                           DataFlowSolver *solver) {
  if (genericOp.getBody()->getOperations().size() > 2) {
    // Each secret.generic should contain at most one instruction -
    // secret-distribute-generic can be used to distribute through the
    // arithmetic ops.
    return false;
  }

  auto &innerOp = genericOp.getBody()->getOperations().front();
  if (!isa<mlir::linalg::MatmulOp>(innerOp)) {
    return false;
  }

  // Determine if the left or right operand is secret to determine which
  // matrix to diagonalize, or if both are secret or both are public, then
  // return failure.
  mlir::linalg::MatmulOp op = cast<mlir::linalg::MatmulOp>(innerOp);
  bool isLeftOperandSecret = isSecret(op.getInputs()[0], solver);
  bool isRightOperandSecret = isSecret(op.getInputs()[1], solver);

  // Error out if both are secret or both are public
  if ((isLeftOperandSecret && isRightOperandSecret) ||
      (!isLeftOperandSecret && !isRightOperandSecret)) {
    return false;
  }
  return true;
}

struct SecretGenericOpLinalgMatmulConversion
    : public OpConversionPattern<secret::GenericOp> {
 private:
  DataFlowSolver *solver;
  int maxTilingSize;

 public:
  using OpConversionPattern<secret::GenericOp>::OpConversionPattern;

  SecretGenericOpLinalgMatmulConversion(const TypeConverter &converter,
                                        DataFlowSolver *solver,
                                        mlir::MLIRContext *context,
                                        int maxTilingSize)
      : OpConversionPattern<secret::GenericOp>(converter, context),
        solver(solver),
        maxTilingSize(maxTilingSize) {}

  LogicalResult matchAndRewrite(
      secret::GenericOp genericOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSquatPackableMatmul(genericOp, solver)) {
      return failure();
    }

    mlir::linalg::MatmulOp op = cast<mlir::linalg::MatmulOp>(
        genericOp.getBody()->getOperations().front());
    bool isLeftOperandSecret = isSecret(op.getInputs()[0], solver);
    bool isRightOperandSecret = isSecret(op.getInputs()[1], solver);

    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();

    // Assign if the left operand is secret
    int secretValuesIndex = 0;
    Value publicMatrix = inputs[1];
    if (isRightOperandSecret) {
      publicMatrix = inputs[0];
      secretValuesIndex = 1;
    }
    auto matrixTensorType = cast<RankedTensorType>(publicMatrix.getType());
    auto bias = outputs[0];

    auto dimensions = matrixTensorType.getShape();
    int64_t dim0 = dimensions[0];  // This is the number of rows in the
    // matrix
    int64_t dim1 = dimensions[1];  // This is the number of columns
                                   // in the matrix.

    // If one of these dimensions is not a power of two, then we can't do
    // the Halevi-Shoup or Squat Packing Matrix Multiplication conversion.    if
    if (!isPowerOfTwo(dim0) || !isPowerOfTwo(dim1)) {
      return failure();
    }

    // Diagonalize the matrix only if the matrix is a constant.
    auto matrixConstantValues =
        dyn_cast<arith::ConstantOp>(publicMatrix.getDefiningOp());
    auto biasConstantValues = dyn_cast<arith::ConstantOp>(bias.getDefiningOp());
    if (!matrixConstantValues || !biasConstantValues) {
      return failure();
    }

    DenseElementsAttr denseAttr =
        dyn_cast<DenseElementsAttr>(matrixConstantValues.getValueAttr());
    DenseElementsAttr biasAttr =
        dyn_cast<DenseElementsAttr>(biasConstantValues.getValueAttr());

    // If the constant values doesn't have a dense attribute, then we can't
    // diagonalize the matrix.
    if (!denseAttr || !biasAttr) {
      return failure();
    }

    auto type = denseAttr.getElementType();
    auto originalShape = denseAttr.getType().getShape();

    if (!type.isIntOrFloat()) {
      return failure();
    }

    // Define local function pointers or lambdas that refer to the functions
    auto diagMatrixInt = diagonalizeMatrix<APInt>;
    auto duplicateBiasInt = duplicateBias<APInt>;
    auto multDiagMatrixWithVectorInt =
        multiplyDiagonalizedMatrixWithVector<arith::AddIOp, arith::MulIOp>;

    auto diagMatrixFloat = diagonalizeMatrix<APFloat>;
    auto duplicateBiasFloat = duplicateBias<APFloat>;
    auto multDiagMatrixWithVectorFloat =
        multiplyDiagonalizedMatrixWithVector<arith::AddFOp, arith::MulFOp>;

    SmallVector<Value> genericOpInputs;
    for (OpOperand &operand : op->getOpOperands()) {
      if (auto *secretArg =
              genericOp.getOpOperandForBlockArgument(operand.get())) {
        genericOpInputs.push_back(
            adaptor.getODSOperands(0)[secretArg->getOperandNumber()]);
      } else {
        genericOpInputs.push_back(operand.get());
      }
    }

    SmallVector<Type> genericOpOutputTypes;
    auto result = getTypeConverter()->convertTypes(genericOp.getResultTypes(),
                                                   genericOpOutputTypes);
    if (failed(result)) return failure();

    auto newGeneric = rewriter.create<secret::GenericOp>(
        genericOp.getLoc(), genericOpInputs, genericOpOutputTypes,
        [&](OpBuilder &builder, Location loc, ValueRange blockArguments) {
          // The blockArguments should include the secret vector and public
          // matrix.

          Value secretValues = blockArguments[secretValuesIndex];
          ImplicitLocOpBuilder b(loc, rewriter);
          Value result;

          // Compute diagonalized matrix and duplicated bias inside the body.
          if (type.isInteger()) {
            Value diagonalizedMatrix =
                diagMatrixInt(b, denseAttr, isLeftOperandSecret, maxTilingSize);
            Value duplicatedBias = duplicateBiasInt(
                b, biasAttr, isLeftOperandSecret, maxTilingSize);
            result = multDiagMatrixWithVectorInt(
                b, diagonalizedMatrix, originalShape, secretValues,
                duplicatedBias, isLeftOperandSecret, maxTilingSize);
          } else {  // Floating point
            Value diagonalizedMatrix = diagMatrixFloat(
                b, denseAttr, isLeftOperandSecret, maxTilingSize);
            Value duplicatedBias = duplicateBiasFloat(
                b, biasAttr, isLeftOperandSecret, maxTilingSize);
            result = multDiagMatrixWithVectorFloat(
                b, diagonalizedMatrix, originalShape, secretValues,
                duplicatedBias, isLeftOperandSecret, maxTilingSize);
          }
          // Yield the final result.
          b.create<secret::YieldOp>(loc, result);
        });

    // Replace the original operation with the new genericOp
    rewriter.replaceOp(genericOp, newGeneric);
    return success();
  }
};

struct LinalgToTensorExt
    : public impl::LinalgToTensorExtBase<LinalgToTensorExt> {
  using LinalgToTensorExtBase::LinalgToTensorExtBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ConversionTarget target(*context);

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

    // TODO: loop through all of the secret values, figure out the tiling size.
    // For now, take tilingSize as a command line argument.

    ReplicatedTensorTypeConverter replicatedTypeConverter(tilingSize);
    RewritePatternSet patterns(context);

    patterns.add<SecretGenericOpLinalgMatmulConversion>(
        replicatedTypeConverter, &solver, context, tilingSize);
    target.addDynamicallyLegalOp<secret::GenericOp>([&](secret::GenericOp op) {
      return !isSquatPackableMatmul(op, &solver);
    });

    addStructuralConversionPatterns(replicatedTypeConverter, patterns, target);
    // Override the default function legality checks to ensure that we only
    // upgrade types when matmul packing is needed. In the future, an analysis
    // pass will be used to  determine function argument packing.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool valueUsedInMatmul = false;
      for (auto value : op.getFunctionBody().getArguments()) {
        if (auto secretTy = dyn_cast<secret::SecretType>(value.getType())) {
          for (auto use : value.getUsers()) {
            if (auto genericOp = dyn_cast<secret::GenericOp>(use)) {
              if (isSquatPackableMatmul(genericOp, &solver)) {
                valueUsedInMatmul = true;
                break;
              }
            }
          }
        }
      }
      if (valueUsedInMatmul) {
        return replicatedTypeConverter.isSignatureLegal(op.getFunctionType()) &&
               replicatedTypeConverter.isLegal(&op.getBody());
      }
      return true;
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      auto valueValid = validateValues(op, [&](Value value) {
        if (auto secretTy = dyn_cast<secret::SecretType>(value.getType())) {
          if (auto genericOp =
                  dyn_cast_or_null<secret::GenericOp>(value.getDefiningOp())) {
            if (isSquatPackableMatmul(genericOp, &solver)) {
              return failure();
            }
          }
        }
        return success();
      });
      if (failed(valueValid)) {
        return replicatedTypeConverter.isLegal(op);
      }
      return true;
    });

    // Run pattern matching and conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace linalg
}  // namespace heir
}  // namespace mlir
