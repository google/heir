#ifndef LIB_UTILS_POLYNOMIAL_RNSPOLYNOMIAL_H_
#define LIB_UTILS_POLYNOMIAL_RNSPOLYNOMIAL_H_

#include <optional>

#include "lib/Dialect/Polynomial/IR/PolynomialEnums.h"
#include "lib/Dialect/RNS/IR/RNSAttributes.h"
#include "llvm/include/llvm/ADT/APInt.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

/// A class representing a polynomial in Residue Number System (RNS) form.
///
/// An RNS polynomial is represented by its residues modulo a set of coprime
/// moduli (limbs). For a polynomial of degree < N and an RNS basis of k moduli
/// [q_0, ..., q_{k-1}], the polynomial is stored as k independent polynomials
/// (limbs), where the i-th limb is a polynomial with coefficients modulo q_i.
///
/// The coefficients of all limbs are stored in a single flat array (`data`)
/// in limb-major order:
///   [limb_0_coeff_0, limb_0_coeff_1, ..., limb_0_coeff_{N-1},
///    limb_1_coeff_0, ..., limb_{k-1}_coeff_{N-1}]
///
/// An RNS polynomial can be in either Coefficient representation (storing the
/// polynomial coefficients directly) or NTT representation (storing the
/// evaluations/slots of the polynomial at the roots of unity). Arithmetic
/// operations (add, sub) are only valid between polynomials in the same
/// representation.
class RNSPolynomial {
 public:
  RNSPolynomial() = default;
  RNSPolynomial(llvm::SmallVector<uint64_t> data,
                llvm::SmallVector<uint64_t> moduli,
                Form representation = Form::COEFF);

  /// Returns the flat data array.
  llvm::ArrayRef<uint64_t> getData() const { return data; }

  /// Returns the moduli (RNS basis).
  llvm::ArrayRef<uint64_t> getModuli() const { return moduli; }

  /// Returns the number of limbs (moduli) in the RNS basis.
  size_t getNumLimbs() const { return moduli.size(); }

  /// Returns the number of coefficients (or slots) per limb.
  unsigned getNumCoeffs() const { return numCoeffs; }

  /// Returns the representation form of the polynomial.
  Form getRepresentation() const { return representation; }

  /// Returns true if the polynomial is in NTT representation.
  bool isNtt() const { return representation == Form::EVAL; }

  /// Returns the element (coefficient or evaluation slot) at the given limb
  /// and coefficient index.
  uint64_t getElement(size_t limbIdx, size_t coeffIdx) const {
    return data[limbIdx * numCoeffs + coeffIdx];
  }

  /// Performs modular addition limb-wise.
  RNSPolynomial add(const RNSPolynomial& other) const;

  /// Performs modular subtraction limb-wise.
  RNSPolynomial sub(const RNSPolynomial& other) const;

  /// Performs modular scalar multiplication limb-wise.
  std::optional<RNSPolynomial> scalarMul(
      llvm::ArrayRef<uint64_t> scalarValues) const;

  /// Performs modular multiplication limb-wise. In NTT form, this corresponds
  /// to an elementwise product. In coefficient form, it first converts to NTT
  /// form, multiplies elementwise, and converts back to coefficient form.
  RNSPolynomial mul(const RNSPolynomial& other) const;

  /// Convert the polynomial to NTT representation.
  RNSPolynomial toNtt(llvm::ArrayRef<uint64_t> rootOfUnity) const;
  RNSPolynomial toNtt(rns::RNSAttr rootAttr = nullptr) const;

  /// Convert the polynomial to Coefficient representation.
  RNSPolynomial toCoefficient(llvm::ArrayRef<uint64_t> rootOfUnity) const;
  RNSPolynomial toCoefficient(rns::RNSAttr rootAttr = nullptr) const;

  /// Slice the polynomial's RNS basis.
  RNSPolynomial slice(size_t start, size_t size) const;

  bool operator==(const RNSPolynomial& other) const {
    return data == other.data && moduli == other.moduli &&
           numCoeffs == other.numCoeffs &&
           representation == other.representation;
  }
  bool operator!=(const RNSPolynomial& other) const {
    return !(*this == other);
  }

 private:
  /// Flat array holding the polynomial data.
  /// In Coefficient form, this holds the coefficients in limb-major order.
  /// In NTT form, this holds the evaluations (slots) in limb-major order.
  llvm::SmallVector<uint64_t> data;

  /// The moduli (RNS basis).
  llvm::SmallVector<uint64_t> moduli;

  unsigned numCoeffs = 0;

  /// The representation form of the polynomial.
  Form representation = Form::COEFF;
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_RNSPOLYNOMIAL_H_
