#include "lib/Dialect/BGV/IR/BGVOps.h"

#include <optional>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() { return lwe::verifyMulOp(this); }

LogicalResult MulPlainOp::verify() { return lwe::verifyMulPlainOp(this); }

LogicalResult RotateColumnsOp::verify() { return lwe::verifyRotateOp(this); }

LogicalResult RotateRowsOp::verify() { return lwe::verifyRotateOp(this); }

LogicalResult RelinearizeOp::verify() { return lwe::verifyRelinearizeOp(this); }

LogicalResult ModulusSwitchOp::verify() {
  return lwe::verifyModulusSwitchOrRescaleOp(this);
}

LogicalResult ExtractOp::verify() { return lwe::verifyExtractOp(this); }

LogicalResult LevelReduceOp::verify() { return lwe::verifyLevelReduceOp(this); }

//===----------------------------------------------------------------------===//
// Op type inference.
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, AddOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult AddPlainOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, AddPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult SubOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, SubOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult SubPlainOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, SubPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, MulOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferMulOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult MulPlainOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, MulPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferMulPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RelinearizeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, RelinearizeOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return lwe::inferRelinearizeOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

void MulPlainOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<lwe::PutCiphertextInFirstOperand<MulPlainOp>>(context);
}
void AddPlainOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<lwe::PutCiphertextInFirstOperand<AddPlainOp>>(context);
}

}  // namespace bgv
}  // namespace heir
}  // namespace mlir
