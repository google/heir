#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"

#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"

void RotateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  populateWithGenerated(results);
}

LogicalResult RotateOp::verify() {
  auto x = getTensor().getType();
  // TODO(#924): Currently RotateOp only supports rotating a 1-D vector, or a
  // vector with only one non-unit dimension that is treated as the major
  // dimension.
  if (x.getRank() != 1) {
    if (llvm::count_if(x.getShape(), [](auto dim) { return dim != 1; }) != 1) {
      return emitOpError() << "requires a 1-D input tensor or tensor with "
                              "single non-unit dimension, but found "
                           << x;
    }
  }
  return success();
}

LogicalResult ConvertLayoutOp::verify() {
  int64_t rank = cast<RankedTensorType>(getTensor().getType()).getRank();
  const AffineMap &fromLayout = getFromLayout().getValue();
  const AffineMap &toLayout = getToLayout().getValue();

  if (rank != fromLayout.getNumDims() || rank != toLayout.getNumDims()) {
    std::string fromLayoutStr, toLayoutStr;
    llvm::raw_string_ostream fromLayoutStream(fromLayoutStr),
        toLayoutStream(toLayoutStr);
    fromLayout.print(fromLayoutStream);
    toLayout.print(toLayoutStream);

    return emitOpError()
           << "requires tensor rank to match the layout map's dimension count"
              "but found rank "
           << rank << " and maps " << fromLayoutStream.str() << " and "
           << toLayoutStream.str();
  }

  return success();
}

LogicalResult SumOp::verify() {
  auto inputTensor = cast<RankedTensorType>(getTensor().getType());
  auto outputTensor = cast<RankedTensorType>(getOutput().getType());

  if (inputTensor.getElementType() != outputTensor.getElementType()) {
    return emitOpError()
           << "requires input and output tensors to have the same "
              "element type, but found "
           << inputTensor.getElementType() << " and "
           << outputTensor.getElementType();
  }

  // The input and output must have the same shape when removing the index
  // given by the operand dim.
  unsigned int dim = getDim().getZExtValue();
  SmallVector<int64_t, 4> inputShape;
  for (int i = 0; i < inputTensor.getRank(); i++) {
    if (i == dim) continue;
    inputShape.push_back(inputTensor.getShape()[i]);
  }

  ArrayRef<int64_t> outputShape = outputTensor.getShape();

  if (llvm::any_of(llvm::zip(inputShape, outputShape), [](auto pair) {
        return std::get<0>(pair) != std::get<1>(pair);
      })) {
    return emitOpError()
           << "requires input and output tensors to have the same shape, but "
              "after summing along dimension "
           << dim << " the input shape becomes " << inputTensor.getShape()
           << " but the output shape is " << outputTensor.getShape();
  }

  return success();
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
