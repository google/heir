#include "lib/Dialect/LWE/IR/LWEOps.h"

#include <cassert>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

OpFoldResult MulScalarOp::fold(MulScalarOp::FoldAdaptor adaptor) {
  if (matchPattern(adaptor.getScalar(), m_One())) {
    return getCiphertext();
  }

  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

LogicalResult RMulOp::verify() { return lwe::verifyMulOp(this); }

LogicalResult RMulPlainOp::verify() { return lwe::verifyMulPlainOp(this); }

LogicalResult TrivialEncryptOp::verify() {
  auto applicationData = this->getInput().getType().getApplicationData();
  auto outApplicationData = this->getOutput().getType().getApplicationData();

  if (applicationData != outApplicationData) {
    return this->emitOpError()
           << "application data of the input and output must match, but "
           << "found input attr " << applicationData << " and output attr "
           << outApplicationData;
  }

  auto plaintextSpace = this->getInput().getType().getPlaintextSpace();
  auto outPlaintextSpace = this->getOutput().getType().getPlaintextSpace();

  if (plaintextSpace != outPlaintextSpace) {
    return this->emitOpError()
           << "plaintext space of the input and output must match, but "
           << "found input attr " << plaintextSpace << " and output attr "
           << outPlaintextSpace;
  }

  auto outCiphertextModulus = this->getOutput()
                                  .getType()
                                  .getCiphertextSpace()
                                  .getRing()
                                  .getCoefficientType()
                                  .getIntOrFloatBitWidth();
  if (outCiphertextModulus != this->getCiphertextBits().getZExtValue()) {
    return this->emitOpError()
           << "ciphertext modulus of the output must match the ciphertext_bits "
              "parameter, expected "
           << this->getCiphertextBits().getZExtValue() << " but found "
           << outCiphertextModulus;
  }

  return success();
}

LogicalResult ReinterpretApplicationDataOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (inputType.getPlaintextSpace() != outputType.getPlaintextSpace() ||
      inputType.getCiphertextSpace() != outputType.getCiphertextSpace() ||
      inputType.getKey() != outputType.getKey() ||
      inputType.getModulusChain() != outputType.getModulusChain()) {
    return emitOpError()
           << "the only allowed difference in the input and output are in the "
              "application_data field, but found input type "
           << inputType << " and output type " << outputType;
  }

  return success();
}

// Verification for RLWE_EncryptOp
LogicalResult RLWEEncryptOp::verify() {
  Type keyType = getKey().getType();
  auto keyRing =
      llvm::TypeSwitch<Type, mlir::heir::polynomial::RingAttr>(keyType)
          .Case<lwe::NewLWEPublicKeyType, lwe::NewLWESecretKeyType>(
              [](auto key) { return key.getRing(); })
          .Default([](Type) {
            llvm_unreachable("impossible by type constraints");
            return nullptr;
          });

  auto outputRing = getOutput().getType().getCiphertextSpace().getRing();
  if (outputRing != keyRing) {
    return emitOpError() << "RLWEEncryptOp rings do not match. Key ring: "
                         << keyRing
                         << ". Output ciphertext ring: " << outputRing << ".";
  }
  return success();
}

// Verify Encoding and Type match
LogicalResult verifyEncodingAndTypeMatch(mlir::Type type,
                                         mlir::Attribute encoding) {
  // En/Decode Ops only allow IntegerOrFloatLike (-> assert not if)
  assert(getElementTypeOrSelf(type).isIntOrFloat() &&
         "Encoding Ops only allow IntegerOrFloatLike types");

  // Verification conditions for each encoding we have:

  if (isa<FullCRTPackingEncodingAttr>(encoding)) {
    // also supports lists of integers and scalars via replication
    return success(getElementTypeOrSelf(type).isInteger());
  }

  if (isa<InverseCanonicalEncodingAttr>(encoding)) {
    // CKKS-style Encoding should support everything
    // (ints via cast to float/double, scalars via replication)
    return success();
  }

  // This code should never be hit unless we added an encoding and forgot to
  // update this function. Assert(false) for DEBUG, return failure for NDEBUG.
  assert(false && "Encoding not handled in encode/decode verifier.");
  return failure();
}

LogicalResult EncodeOp::verify() {
  auto plaintextType = getOutput().getType();

  // Output type must have application data that matches the input type.
  auto applicationDataTy = plaintextType.getApplicationData().getMessageType();
  if (applicationDataTy != getInput().getType()) {
    return emitOpError()
           << "output type application data must match input type, expected "
           << getInput().getType() << " but got " << applicationDataTy;
  }

  // LWE plaintext types must have plaintext ring modulus f(x) = x and
  // coefficient type matching input message bits parameter.
  auto plaintextRing = plaintextType.getPlaintextSpace().getRing();

  auto polyTerms =
      plaintextRing.getPolynomialModulus().getPolynomial().getTerms();
  if (polyTerms.size() != 1) {
    return emitOpError()
           << "LWE plaintext ring modulus must have exactly one term";
  }
  const auto& firstTerm = polyTerms[0];
  if (firstTerm.getCoefficient() != 1 && firstTerm.getExponent() != 1) {
    return emitOpError() << "LWE plaintext ring modulus must be x";
  }
  auto integerTy = dyn_cast<IntegerType>(plaintextRing.getCoefficientType());
  if (!integerTy) {
    return emitOpError()
           << "LWE plaintext ring coefficient type must be integer";
  }
  if (integerTy.getWidth() != getPlaintextBits().getZExtValue()) {
    return emitOpError()
           << "LWE plaintext ring coefficient type width must match message "
              "bits parameter, expected "
           << getPlaintextBits().getZExtValue() << " but got "
           << integerTy.getWidth();
  }

  // Overflow attr must matches overflow parameter in output type.
  if (getOverflow() != plaintextType.getApplicationData().getOverflow()) {
    return emitOpError()
           << "output type overflow must match overflow parameter, expected "
           << getOverflow() << " but got "
           << plaintextType.getApplicationData().getOverflow();
  }
  return success();
}

LogicalResult RLWEEncodeOp::verify() {
  return verifyEncodingAndTypeMatch(getInput().getType(), getEncoding());
}

LogicalResult RLWEDecodeOp::verify() {
  return verifyEncodingAndTypeMatch(getResult().getType(), getEncoding());
}

//===----------------------------------------------------------------------===//
// Op type inference.
//===----------------------------------------------------------------------===//

LogicalResult RAddOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RAddOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RAddPlainOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RAddPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RSubOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RSubOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RSubPlainOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RSubPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RMulOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RMulOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferMulOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

LogicalResult RMulPlainOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RMulPlainOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return lwe::inferMulPlainOpReturnTypes(ctx, adaptor, inferredReturnTypes);
}

void RAddPlainOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<lwe::PutCiphertextInFirstOperand<RAddPlainOp>>(context);
}

void RMulPlainOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<lwe::PutCiphertextInFirstOperand<RMulPlainOp>>(context);
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
