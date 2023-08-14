#include "include/Dialect/Poly/IR/PolyOps.h"

namespace mlir {
namespace heir {
namespace poly {

LogicalResult ExtractSliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ExtractSliceOpAdaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  PolynomialType polyType =
      llvm::dyn_cast<PolynomialType>(adaptor.getInput().getType());
  RingAttr attr = polyType.getRing();
  uint32_t idealDegree = attr.ideal().getDegree();
  IntegerType elementTy =
      IntegerType::get(context, attr.coefficientModulus().getBitWidth(),
                       IntegerType::SignednessSemantics::Unsigned);
  Type resultType = RankedTensorType::get({idealDegree}, elementTy, attr);
  inferredReturnTypes.assign({resultType});
  return success();
}

void PolyFromCoeffsOp::build(OpBuilder &builder, OperationState &result,
                             Value input, RingAttr ring) {
  TensorType tensorType = dyn_cast<TensorType>(input.getType());
  APInt cmod(APINT_BIT_WIDTH, 1);
  cmod = cmod << tensorType.getElementTypeBitWidth();
  Type resultType = PolynomialType::get(builder.getContext(), ring);
  build(builder, result, resultType, input);
}

}  // namespace poly
}  // namespace heir
}  // namespace mlir
