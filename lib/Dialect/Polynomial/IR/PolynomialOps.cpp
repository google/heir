#include "include/Dialect/Polynomial/IR/PolynomialOps.h"

#include "llvm/include/llvm/ADT/APInt.h"               // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

// Required after PatternMatch.h
#include "include/Dialect/Polynomial/IR/PolynomialCanonicalize.cpp.inc"

void FromTensorOp::build(OpBuilder &builder, OperationState &result,
                         Value input, RingAttr ring) {
  TensorType tensorType = dyn_cast<TensorType>(input.getType());
  auto bitWidth = tensorType.getElementTypeBitWidth();
  APInt cmod(1 + bitWidth, 1);
  cmod = cmod << bitWidth;
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

static LogicalResult verifyNTTOp(Operation *op, RingAttr ring,
                                 RankedTensorType tensorType) {
  auto encoding = tensorType.getEncoding();
  if (!encoding) {
    return op->emitOpError()
           << "a ring encoding was not provided to the tensor.";
  }
  auto encodedRing = dyn_cast<RingAttr>(encoding);
  if (!encodedRing) {
    return op->emitOpError()
           << "the provided tensor encoding is not a ring attribute.";
  }

  if (encodedRing != ring) {
    return op->emitOpError()
           << "encoded ring type " << encodedRing
           << " is not equivalent to the polynomial ring " << ring << ".";
  }

  auto polyDegree = ring.getIdeal().getDegree();
  auto tensorShape = tensorType.getShape();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] == polyDegree;
  if (!compatible) {
    return op->emitOpError()
           << "tensor type " << tensorType
           << " must be a tensor of shape [d] where d "
           << "is exactly the degree of the ring's ideal " << ring;
  }

  if (!ring.primitive2NthRoot()) {
    return op->emitOpError()
           << "ring type does not provide a primitive root 2n-th primitive root"
           << "of unity, where n is the polynomial degree: " << polyDegree;
  }

  return success();
}

LogicalResult NTTOp::verify() {
  auto ring = getInput().getType().getRing();
  auto tensorType = getOutput().getType();
  return verifyNTTOp(this->getOperation(), ring, tensorType);
}

LogicalResult INTTOp::verify() {
  auto tensorType = getInput().getType();
  auto ring = getOutput().getType().getRing();
  return verifyNTTOp(this->getOperation(), ring, tensorType);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  populateWithGenerated(results);
}

void NTTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  populateWithGenerated(results);
}

void INTTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  populateWithGenerated(results);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
