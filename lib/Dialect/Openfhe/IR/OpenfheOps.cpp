#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"

#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

LogicalResult MulNoRelinOp::verify() { return lwe::verifyMulOp(this); }

LogicalResult MakePackedPlaintextOp::verify() {
  auto enc = this->getPlaintext().getType().getPlaintextSpace().getEncoding();
  if (!llvm::isa<lwe::FullCRTPackingEncodingAttr>(enc)) {
    return emitOpError("plaintext type should use full_crt_packing_encoding.");
  }
  return success();
}

LogicalResult MakeCKKSPackedPlaintextOp::verify() {
  auto enc = this->getPlaintext().getType().getPlaintextSpace().getEncoding();
  if (!llvm::isa<lwe::InverseCanonicalEncodingAttr>(enc)) {
    return emitOpError("plaintext type should use inverse_canonical_encoding.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Op type inference.
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, AddOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult AddPlainOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, AddPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult SubOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, SubOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult SubPlainOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, SubPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult MulNoRelinOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, MulNoRelinOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferMulOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
