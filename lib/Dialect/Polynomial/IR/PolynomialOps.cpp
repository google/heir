#include "include/Dialect/Polynomial/IR/PolynomialOps.h"

#include "llvm/include/llvm/ADT/APInt.h"               // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

// Required after PatternMatch.h
#include "include/Dialect/Polynomial/IR/PolynomialCanonicalize.cpp.inc"

namespace mlir {
namespace heir {
namespace polynomial {

void FromTensorOp::build(OpBuilder &builder, OperationState &result,
                         Value input, RingAttr ring) {
  TensorType tensorType = dyn_cast<TensorType>(input.getType());
  APInt cmod(APINT_BIT_WIDTH, 1);
  cmod = cmod << tensorType.getElementTypeBitWidth();
  Type resultType = PolynomialType::get(builder.getContext(), ring);
  build(builder, result, resultType, input);
}

LogicalResult FromTensorOp::verify() {
  auto tensorShape = getInput().getType().getShape();
  auto ring = getOutput().getType().getRing();
  auto polyDegree = ring.getIdeal().getDegree();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] <= polyDegree;
  if (!compatible) {
    return emitOpError()
           << "input type " << getInput().getType()
           << " does not match output type " << getOutput().getType()
           << ". The input type must be a tensor of shape [d] where d "
              "is at most the degree of the polynomial generator of "
              "the output ring's ideal.";
  }

  APInt coefficientModulus = ring.coefficientModulus();
  unsigned cmodBitWidth = coefficientModulus.ceilLogBase2();
  unsigned inputBitWidth = getInput().getType().getElementTypeBitWidth();

  if (inputBitWidth > cmodBitWidth) {
    return emitOpError() << "input tensor element type "
                         << getInput().getType().getElementType()
                         << " is too large to fit in the coefficients of "
                         << getOutput().getType()
                         << ". The input tensor's elements must be rescaled"
                            " to fit before using from_tensor.";
  }

  return success();
}

LogicalResult ToTensorOp::verify() {
  auto tensorShape = getOutput().getType().getShape();
  auto polyDegree = getInput().getType().getRing().getIdeal().getDegree();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;

  return compatible
             ? success()
             : emitOpError()
                   << "input type " << getInput().getType()
                   << " does not match output type " << getOutput().getType()
                   << ". The input type must be a tensor of shape [d] where d "
                      "is exactly the degree of the polynomial generator of "
                      "the output ring's ideal.";
}

LogicalResult MonomialMulOp::verify() {
  auto ring = getInput().getType().getRing();
  auto idealTerms = ring.getIdeal().getTerms();
  bool compatible =
      idealTerms.size() == 2 &&
      (idealTerms[0].coefficient == -1 && idealTerms[0].exponent == 0) &&
      (idealTerms[1].coefficient == 1);

  return compatible ? success()
                    : emitOpError()
                          << "ring type " << ring
                          << " is not supported yet. The ring "
                             "must be of the form (x^n - 1) for some n";
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  populateWithGenerated(results);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
