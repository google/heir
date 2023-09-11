#include "include/Dialect/Poly/IR/PolyOps.h"

namespace mlir {
namespace heir {
namespace poly {

void FromTensorOp::build(OpBuilder &builder, OperationState &result,
                         Value input, RingAttr ring) {
  TensorType tensorType = dyn_cast<TensorType>(input.getType());
  APInt cmod(APINT_BIT_WIDTH, 1);
  cmod = cmod << tensorType.getElementTypeBitWidth();
  Type resultType = PolyType::get(builder.getContext(), ring);
  build(builder, result, resultType, input);
}

LogicalResult FromTensorOp::verify() {
  auto tensorShape = getInput().getType().getShape();
  auto polyDegree = getOutput().getType().getRing().getIdeal().getDegree();
  bool compatible = tensorShape.size() == 1 && tensorShape[0] <= polyDegree;
  return compatible
             ? success()
             : emitOpError()
                   << "input type " << getInput().getType()
                   << " does not match output type " << getOutput().getType()
                   << ". The input type must be a tensor of shape [d] where d "
                      "is at most the degree of the polynomial generator of "
                      "the output ring's ideal.";
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

}  // namespace poly
}  // namespace heir
}  // namespace mlir
