#include "lib/Dialect/BGV/IR/BGVOps.h"

#include "lib/Dialect/LWE/IR/LWEOps.h"

namespace mlir {
namespace heir {
namespace bgv {

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

// TODO: verify scaling factor for add/mul

LogicalResult MulOp::verify() { return lwe::verifyMulOp(this); }

LogicalResult RotateColumnsOp::verify() { return lwe::verifyRotateOp(this); }

LogicalResult RotateRowsOp::verify() { return lwe::verifyRotateOp(this); }

LogicalResult RelinearizeOp::verify() { return lwe::verifyRelinearizeOp(this); }

LogicalResult ModulusSwitchOp::verify() {
  return lwe::verifyModulusSwitchOrRescaleOp(this);
}

LogicalResult ExtractOp::verify() { return lwe::verifyExtractOp(this); }

//===----------------------------------------------------------------------===//
// Op type inference.
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, AddOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult SubOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, SubOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, MulOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferMulOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RelinearizeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, RelinearizeOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferRelinearizeOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
