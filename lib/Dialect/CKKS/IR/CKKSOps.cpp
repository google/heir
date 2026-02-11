#include "lib/Dialect/CKKS/IR/CKKSOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
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
  auto ringEltType = cast<lwe::LWERingEltType>(op.getValue().getType());
  lwe::LWERingEltType outRingType =
      lwe::LWERingEltType::get(ctx, ringEltType.getRing());
  results.push_back(outRingType);
  results.push_back(outRingType);
  return success();
}

LogicalResult KeySwitchInnerOp::verify() {
  RankedTensorType keyTensorType = getKeySwitchingKey().getType();
  auto ctType =
      dyn_cast<lwe::LWECiphertextType>(keyTensorType.getElementType());
  if (!ctType) {
    return emitOpError() << "KSKs must be a tensor of Ciphertexts";
  }
  polynomial::RingAttr ringType = ctType.getCiphertextSpace().getRing();
  auto keyRNSType = dyn_cast<rns::RNSType>(ringType.getCoefficientType());
  if (!keyRNSType) {
    return emitOpError() << "Keyswitch key must be a ring element of RNS types";
  }

  auto ringEltType = cast<lwe::LWERingEltType>(getValue().getType());
  auto inputRNSType =
      dyn_cast<rns::RNSType>(ringEltType.getRing().getCoefficientType());
  if (!inputRNSType) {
    return emitOpError() << "Value must be a ring element of RNS types";
  }

  int kskRank = keyTensorType.getRank();
  if (kskRank != 1) {
    return emitOpError()
           << "KeySwitchingKey must be a rank-1 tensor, but it has rank  "
           << kskRank;
  }

  SchemeParamAttr schemeParamAttr =
      getOperation()
          ->getParentOfType<ModuleOp>()
          ->getAttrOfType<SchemeParamAttr>(CKKSDialect::kSchemeParamAttrName);
  if (!schemeParamAttr) {
    return emitOpError()
           << "Cannot find scheme param attribute on parent module";
  }
  auto schemeParam = getSchemeParamFromAttr(schemeParamAttr);

  SmallVector<Type> extModuli;
  Builder b(getContext());
  for (auto ty : inputRNSType.getBasisTypes()) {
    extModuli.push_back(ty);
  }
  for (auto prime : schemeParam.getPi()) {
    extModuli.push_back(
        mod_arith::ModArithType::get(getContext(), b.getI64IntegerAttr(prime)));
  }
  rns::RNSType expectedKeyRNSType = rns::RNSType::get(getContext(), extModuli);

  if (keyRNSType != expectedKeyRNSType) {
    return emitOpError() << "Key's RNS type " << keyRNSType
                         << " must be the same as the input's RNS type "
                         << inputRNSType << ", plus the key-switch moduli "
                         << schemeParam.getPi();
  }

  int64_t partSize = schemeParam.getPi().size();
  int rnsLength = inputRNSType.getBasisTypes().size();
  int64_t numFullPartitions = rnsLength / partSize;
  int64_t extraPartStart = partSize * numFullPartitions;
  int64_t extraPartSize = rnsLength - extraPartStart;
  int64_t numParts = numFullPartitions + extraPartSize;
  int kskLen = keyTensorType.getShape()[0];
  if (kskLen != numParts) {
    return emitOpError() << "KeySwitchingKey must have shape " << numParts
                         << "xRNS, but it has shape " << kskLen << "xRNS";
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
