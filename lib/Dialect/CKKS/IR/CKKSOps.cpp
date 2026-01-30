#include "lib/Dialect/CKKS/IR/CKKSOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace ckks {

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() { return lwe::verifyMulOp(this); }

LogicalResult MulPlainOp::verify() { return lwe::verifyMulPlainOp(this); }

LogicalResult RotateOp::verify() { return lwe::verifyRotateOp(this); }

LogicalResult RelinearizeOp::verify() { return lwe::verifyRelinearizeOp(this); }

LogicalResult RescaleOp::verify() {
  return lwe::verifyModulusSwitchOrRescaleOp(this);
}

LogicalResult LevelReduceOp::verify() { return lwe::verifyLevelReduceOp(this); }

LogicalResult BootstrapOp::verify() {
  std::optional<int64_t> targetLevel = getTargetLevel();
  if (targetLevel.has_value()) {
    // If a target level is specified, then the result ciphertext must have that
    // many levels.
    lwe::LWECiphertextType outputType = lwe::getCtTy(getOutput());
    if (outputType.getModulusChain().getCurrent() != targetLevel.value()) {
      return emitOpError() << "output ciphertext must have "
                           << targetLevel.value() << " levels but has "
                           << outputType.getModulusChain().getCurrent();
    }
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

LogicalResult MulOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, MulOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferMulOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult MulPlainOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, MulPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferMulPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RelinearizeOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RelinearizeOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferRelinearizeOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult LevelReduceOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, LevelReduceOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferLevelReduceOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult KeySwitchInnerOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  KeySwitchInnerOpAdaptor op(operands, attrs, properties, regions);
  lwe::LWERingEltType ringEltType =
      cast<lwe::LWERingEltType>(op.getValue().getType());
  results.push_back(ringEltType);
  results.push_back(ringEltType);
  return success();
}

LogicalResult ExtractCoeffOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ExtractCoeffOpAdaptor op(operands, attrs, properties, regions);

  lwe::LWECiphertextType ctType =
      cast<lwe::LWECiphertextType>(op.getValue().getType());
  polynomial::RingAttr ringAttr = ctType.getCiphertextSpace().getRing();
  lwe::LWERingEltType outputType =
      lwe::LWERingEltType::get(ctx, ringAttr, ctType.getModulusChain());

  results.push_back(outputType);
  return success();
}

LogicalResult KeySwitchInnerOp::verify() {
  // TODO(#2157): check the ksk's RNS chain extends the value's RNS chain.
  return success();
}

LogicalResult ExtractCoeffOp::verify() {
  int numCTCoeffs = this->getValue().getType().getCiphertextSpace().getSize();
  int idx = this->getIndex().getZExtValue();

  if (idx < 0) {
    return emitOpError() << "index " << idx << " cannot be negative";
  }

  if (idx >= numCTCoeffs) {
    return emitOpError()
           << "index " << idx
           << " must be smaller than the number of ciphertext components "
           << numCTCoeffs;
  }

  return success();
}

LogicalResult FromCoeffsOp::verify() {
  int numCoeffs = this->getCoeffs().size();
  if (numCoeffs < 1) {
    return emitOpError()
           << "Ciphertexts must have at least two components; got "
           << numCoeffs;
  }
  return success();
}

void MulPlainOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<lwe::PutCiphertextInFirstOperand<MulPlainOp>>(context);
}
void AddPlainOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<lwe::PutCiphertextInFirstOperand<AddPlainOp>>(context);
}

// ElementwiseByOperandOpInterface impl

bool RelinearizeOp::operandIsMappable(unsigned operandIndex) {
  // only `input`
  return operandIndex == 0;
}

}  // namespace ckks
}  // namespace heir
}  // namespace mlir
