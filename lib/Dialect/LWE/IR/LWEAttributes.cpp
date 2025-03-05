#include "lib/Dialect/LWE/IR/LWEAttributes.h"

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir {
namespace heir {
namespace lwe {

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

LogicalResult UnspecifiedBitFieldEncodingAttr::verifyEncoding(
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

  return success();
}

LogicalResult ApplicationDataAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::Type messageType, Attribute overflow) {
  if (!mlir::isa<PreserveOverflowAttr, NoOverflowAttr>(overflow)) {
    return emitError() << "overflow must be either preserve_overflow or "
                       << "no_overflow, but found " << overflow << "\n";
  }

  return success();
}

LogicalResult PlaintextSpaceAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::heir::polynomial::RingAttr ring, Attribute encoding) {
  if (mlir::isa<FullCRTPackingEncodingAttr>(encoding)) {
    // For full CRT packing, the ring must be of the form x^n + 1 and the
    // modulus must be 1 mod n.
    auto polyMod = ring.getPolynomialModulus();
    auto poly = polyMod.getPolynomial();
    auto polyTerms = poly.getTerms();
    if (polyTerms.size() != 2) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    const auto& constantTerm = polyTerms[0];
    const auto& constantCoeff = constantTerm.getCoefficient();
    if (!(constantTerm.getExponent().isZero() && constantCoeff.isOne() &&
          polyTerms[1].getCoefficient().isOne())) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    // Check that the modulus is 1 mod n.
    auto modCoeffTy =
        llvm::dyn_cast<mod_arith::ModArithType>(ring.getCoefficientType());
    if (modCoeffTy) {
      APInt modulus = modCoeffTy.getModulus().getValue();
      unsigned n = poly.getDegree();
      if (!modulus.urem(APInt(modulus.getBitWidth(), n)).isOne()) {
        return emitError()
               << "modulus must be 1 mod n for full CRT packing, mod = "
               << modulus.getZExtValue() << " n = " << n << "\n";
      }
    }
  }

  return success();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
