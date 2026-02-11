#include "lib/Dialect/LWE/IR/LWEOps.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

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

LogicalResult RMulRingEltOp::verify() {
  lwe::LWECiphertextType ct;
  lwe::LWERingEltType pt;
  if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(getLhs()))) {
    ct = getCtTy(getLhs());
    pt = cast<LWERingEltType>(getElementTypeOrSelf(getRhs()));
  } else {
    ct = getCtTy(getRhs());
    pt = cast<LWERingEltType>(getElementTypeOrSelf(getLhs()));
  }
  lwe::LWECiphertextType out = getCtTy(getOutput());
  // verify dimension matches
  if (ct.getCiphertextSpace().getSize() != out.getCiphertextSpace().getSize()) {
    return emitOpError() << "output.dim == x.dim does not hold";
  }
  // verify ring and modulusChain matches
  auto ctCiphertext = ct.getCiphertextSpace();
  if (ctCiphertext.getRing() != pt.getRing()) {
    return emitOpError() << "Input rings do not match";
  }
  return success();
}

LogicalResult TrivialEncryptOp::verify() {
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

// Verification for RLWE_EncryptOp
LogicalResult RLWEEncryptOp::verify() {
  Type keyType = getKey().getType();
  auto keyRing =
      llvm::TypeSwitch<Type, mlir::heir::polynomial::RingAttr>(keyType)
          .Case<lwe::LWEPublicKeyType, lwe::LWESecretKeyType>(
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

  return success();
}

LogicalResult RLWEEncodeOp::verify() {
  if (auto tensorTy = dyn_cast<ShapedType>(getInput().getType())) {
    if (tensorTy.getRank() > 1) {
      return emitOpError() << "RLWEEncodeOp only supports 1D tensors";
    }
  }
  return verifyEncodingAndTypeMatch(getInput().getType(), getEncoding());
}

LogicalResult RLWEDecodeOp::verify() {
  return verifyEncodingAndTypeMatch(getResult().getType(), getEncoding());
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

LogicalResult ExtractSliceOp::verify() {
  auto ringEltType = dyn_cast<lwe::LWERingEltType>(this->getInput().getType());
  if (!ringEltType) return failure();
  auto rnsType =
      dyn_cast<rns::RNSType>(ringEltType.getRing().getCoefficientType());
  if (!rnsType) return failure();
  int64_t start = getStart().getZExtValue();
  int64_t size = getSize().getZExtValue();
  return verifyExtractSliceOp(this, rnsType, start, size);
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

LogicalResult ExtractCoeffOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ExtractCoeffOpAdaptor op(operands, attrs, properties, regions);

  auto ctType = cast<lwe::LWECiphertextType>(op.getValue().getType());
  polynomial::RingAttr ringAttr = ctType.getCiphertextSpace().getRing();
  lwe::LWERingEltType outputType = lwe::LWERingEltType::get(ctx, ringAttr);

  results.push_back(outputType);
  return success();
}

LogicalResult ExtractSliceOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ExtractSliceOpAdaptor op(operands, attrs, properties, regions);
  auto inputType = dyn_cast<lwe::LWERingEltType>(op.getInput().getType());
  if (!inputType) return failure();
  polynomial::RingAttr ringAttr = inputType.getRing();
  auto elementType = dyn_cast<rns::RNSType>(ringAttr.getCoefficientType());
  if (!elementType) return failure();
  rns::RNSType outputRNSType =
      inferExtractSliceReturnTypes(ctx, &op, elementType);
  polynomial::RingAttr outputRingAttr = polynomial::RingAttr::get(
      ctx, outputRNSType, ringAttr.getPolynomialModulus());
  results.push_back(lwe::LWERingEltType::get(ctx, outputRingAttr));
  return success();
}

LogicalResult ConvertBasisOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  ConvertBasisOpAdaptor op(operands, attrs, properties, regions);
  auto inputType = dyn_cast<lwe::LWERingEltType>(op.getValue().getType());
  if (!inputType) return failure();
  polynomial::RingAttr ringAttr = inputType.getRing();
  rns::RNSType elementType = dyn_cast<rns::RNSType>(op.getTargetBasis());
  if (!elementType) return failure();
  polynomial::RingAttr outputRingAttr = polynomial::RingAttr::get(
      ctx, elementType, ringAttr.getPolynomialModulus());
  lwe::LWERingEltType resultType =
      lwe::LWERingEltType::get(ctx, outputRingAttr);
  results.push_back(resultType);
  return success();
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
