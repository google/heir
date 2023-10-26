#include "include/Dialect/LWE/IR/LWEDialect.h"

#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated definitions
#include "include/Dialect/LWE/IR/LWEDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "include/Dialect/LWE/IR/LWEAttributes.cpp.inc"

namespace mlir {
namespace heir {
namespace lwe {

void LWEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "include/Dialect/LWE/IR/LWEAttributes.cpp.inc"
      >();
}

LogicalResult BitFieldEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
  if (!elementType.isSignlessInteger()) {
    return emitError() << "Tensors with a bit_field_encoding must have "
                       << "signless integer element type, but found "
                       << elementType;
  }

  unsigned plaintextBitwidth = elementType.getIntOrFloatBitWidth();
  unsigned cleartextBitwidth = getCleartextBitwidth();
  if (plaintextBitwidth < cleartextBitwidth)
    return emitError() << "The tensor element type's bitwidth "
                       << plaintextBitwidth
                       << " is too small to store the cleartext, "
                       << "which has bit width " << cleartextBitwidth << "";

  auto cleartextStart = getCleartextStart();
  if (cleartextStart < 0 || cleartextStart >= plaintextBitwidth)
    return emitError() << "Attribute's cleartext starting bit index ("
                       << cleartextStart << ") is outside the legal range [0, "
                       << plaintextBitwidth - 1 << "]";

  // It may be worth adding some sort of warning notification if the attribute
  // allocates no bits for noise, since this would be effectively useless for
  // FHE.
  return success();
}

LogicalResult requirePolynomialElementTypeFits(
    Type elementType, llvm::StringRef encodingName, unsigned cleartextBitwidth,
    unsigned cleartextStart,
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (!elementType.isa<polynomial::PolynomialType>()) {
    return emitError() << "Tensors with encoding " << encodingName
                       << " must have `poly.poly` element type, but found "
                       << elementType << "\n";
  }
  polynomial::PolynomialType polyType =
      llvm::cast<polynomial::PolynomialType>(elementType);
  // The coefficient modulus takes the place of the plaintext bitwidth for
  // RLWE.
  unsigned plaintextBitwidth =
      polyType.getRing().coefficientModulus().getBitWidth();

  if (plaintextBitwidth < cleartextBitwidth)
    return emitError() << "The polys in this tensor have a coefficient "
                       << "modulus with bitwidth " << plaintextBitwidth
                       << ", which too small to store the cleartext, "
                       << "which has bit width " << cleartextBitwidth << "";

  if (cleartextStart < 0 || cleartextStart >= plaintextBitwidth)
    return emitError() << "Attribute's cleartext starting bit index ("
                       << cleartextStart << ") is outside the legal range [0, "
                       << plaintextBitwidth - 1 << "]";

  return success();
}

LogicalResult PolynomialCoefficientEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
  return requirePolynomialElementTypeFits(
      elementType, "poly_coefficient_encoding", getCleartextBitwidth(),
      getCleartextStart(), emitError);
}

LogicalResult PolynomialEvaluationEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
  return requirePolynomialElementTypeFits(
      elementType, "poly_evaluation_encoding", getCleartextBitwidth(),
      getCleartextStart(), emitError);
}

LogicalResult InverseCanonicalEmbeddingEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
  return requirePolynomialElementTypeFits(
      elementType, "inverse_canonical_embedding_encoding",
      getCleartextBitwidth(), getCleartextStart(), emitError);
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
