#ifndef LIB_UTILS_POLYNOMIAL_POLYNOMIAL_H_
#define LIB_UTILS_POLYNOMIAL_POLYNOMIAL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "llvm/include/llvm/ADT/APFloat.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/APInt.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/Hashing.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/Twine.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project

namespace mlir {

class MLIRContext;

namespace heir {
namespace polynomial {

/// This restricts statically defined polynomials to have at most 64-bit
/// coefficients. This may be relaxed in the future, but it seems unlikely one
/// would want to specify 128-bit polynomials statically in the source code.
constexpr unsigned apintBitWidth = 64;

template <class Derived, typename CoefficientType>
class MonomialBase {
 public:
  MonomialBase(const CoefficientType& coeff, const APInt& expo)
      : coefficient(coeff), exponent(expo) {}
  virtual ~MonomialBase() = default;

  const CoefficientType& getCoefficient() const { return coefficient; }
  CoefficientType& getMutableCoefficient() { return coefficient; }
  const APInt& getExponent() const { return exponent; }
  void setCoefficient(const CoefficientType& coeff) { coefficient = coeff; }
  void setExponent(const APInt& exp) { exponent = exp; }

  bool operator==(const MonomialBase& other) const {
    return other.coefficient == coefficient && other.exponent == exponent;
  }
  bool operator!=(const MonomialBase& other) const {
    return other.coefficient != coefficient || other.exponent != exponent;
  }

  /// Monomials are ordered by exponent.
  bool operator<(const MonomialBase& other) const {
    return (exponent.ult(other.exponent));
  }

  Derived add(const Derived& other) const {
    assert(exponent == other.exponent);
    CoefficientType newCoeff = coefficient + other.coefficient;
    Derived result;
    result.setCoefficient(newCoeff);
    result.setExponent(exponent);
    return result;
  }

  Derived scale(const CoefficientType& scalar) const {
    CoefficientType newCoeff = coefficient * scalar;
    Derived result;
    result.setCoefficient(newCoeff);
    result.setExponent(APInt(exponent));
    return result;
  }

  virtual bool isMonic() const = 0;

  virtual void coefficientToString(
      llvm::SmallString<16>& coeffString) const = 0;

  template <class D, typename T>
  friend ::llvm::hash_code hash_value(const MonomialBase<D, T>& arg);

 protected:
  CoefficientType coefficient;
  APInt exponent;
};

/// A class representing a monomial of a single-variable polynomial with integer
/// coefficients.
class IntMonomial : public MonomialBase<IntMonomial, APInt> {
 public:
  IntMonomial(int64_t coeff, uint64_t expo)
      : MonomialBase(APInt(apintBitWidth, coeff), APInt(apintBitWidth, expo)) {}

  IntMonomial()
      : MonomialBase(APInt(apintBitWidth, 0), APInt(apintBitWidth, 0)) {}

  ~IntMonomial() override = default;

  bool isMonic() const override { return coefficient == 1; }

  void coefficientToString(llvm::SmallString<16>& coeffString) const override {
    coefficient.toStringSigned(coeffString);
  }
};

/// A class representing a monomial of a single-variable polynomial with integer
/// coefficients.
class FloatMonomial : public MonomialBase<FloatMonomial, APFloat> {
 public:
  FloatMonomial(double coeff, uint64_t expo)
      : MonomialBase(APFloat(coeff), APInt(apintBitWidth, expo)) {}

  FloatMonomial() : MonomialBase(APFloat((double)0), APInt(apintBitWidth, 0)) {}

  ~FloatMonomial() override = default;

  bool isMonic() const override { return coefficient == APFloat(1.0); }

  void coefficientToString(llvm::SmallString<16>& coeffString) const override {
    coefficient.toString(coeffString);
  }
};

template <class Derived, typename Monomial, typename CoefficientType>
class PolynomialBase {
 public:
  PolynomialBase() = delete;
  virtual ~PolynomialBase<Derived, Monomial, CoefficientType>() = default;

  explicit PolynomialBase(ArrayRef<Monomial> terms) : terms(terms) {}

  explicit operator bool() const { return !terms.empty(); }
  bool operator==(const PolynomialBase& other) const {
    return other.terms == terms;
  }
  bool operator!=(const PolynomialBase& other) const {
    return !(other.terms == terms);
  }

  void print(raw_ostream& os, ::llvm::StringRef separator,
             ::llvm::StringRef exponentiation) const {
    bool first = true;
    for (const Monomial& term : getTerms()) {
      if (first) {
        first = false;
      } else {
        os << separator;
      }
      std::string coeffToPrint;
      if (term.isMonic() && term.getExponent().uge(1)) {
        coeffToPrint = "";
      } else {
        llvm::SmallString<16> coeffString;
        term.coefficientToString(coeffString);
        coeffToPrint = coeffString.str();
      }

      if (term.getExponent().isZero()) {
        os << coeffToPrint;
      } else if (term.getExponent().isOne()) {
        os << coeffToPrint << "x";
      } else {
        llvm::SmallString<16> expString;
        term.getExponent().toStringSigned(expString);
        os << coeffToPrint << "x" << exponentiation << expString;
      }
    }
  }

  /// Remove terms with a zero coefficient.
  void canonicalize() {
    for (auto it = terms.begin(); it != terms.end();) {
      if (it->getCoefficient().isZero()) {
        it = terms.erase(it);
      } else {
        ++it;
      }
    }
  }

  Derived add(const Derived& other) const {
    SmallVector<Monomial> newTerms;
    auto it1 = terms.begin();
    auto it2 = other.terms.begin();
    while (it1 != terms.end() || it2 != other.terms.end()) {
      if (it1 == terms.end()) {
        newTerms.emplace_back(*it2);
        it2++;
        continue;
      }

      if (it2 == other.terms.end()) {
        newTerms.emplace_back(*it1);
        it1++;
        continue;
      }

      while (it1->getExponent().ult(it2->getExponent())) {
        newTerms.emplace_back(*it1);
        it1++;
        if (it1 == terms.end()) break;
      }
      if (it1 == terms.end()) continue;

      while (it2->getExponent().ult(it1->getExponent())) {
        newTerms.emplace_back(*it2);
        it2++;
        if (it2 == other.terms.end()) break;
      }
      if (it2 == other.terms.end()) continue;

      Monomial newTerm = it1->add(*it2);
      if (!newTerm.getCoefficient().isZero()) newTerms.push_back(newTerm);

      it1++;
      it2++;
    }
    return Derived(newTerms);
  }

  Derived naiveMul(const Derived& other) const {
    SmallVector<Monomial> newTerms;
    size_t maxDegree = getDegree() + other.getDegree();
    newTerms.reserve(maxDegree + 1);
    for (size_t i = 0; i <= maxDegree; ++i) {
      newTerms.push_back(Monomial(0, i));
    }

    for (size_t i = 0; i < terms.size(); ++i) {
      for (size_t j = 0; j < other.terms.size(); ++j) {
        int combinedDegree = terms[i].getExponent().getZExtValue() +
                             other.terms[j].getExponent().getZExtValue();
        newTerms[combinedDegree].setCoefficient(
            newTerms[combinedDegree].getCoefficient() +
            terms[i].getCoefficient() * other.terms[j].getCoefficient());
      }
    }
    auto result = Derived(newTerms);
    result.canonicalize();
    return result;
  }

  inline DenseMap<int64_t, Monomial> getCoeffMap() const {
    DenseMap<int64_t, Monomial> coeffMap;
    for (auto term : terms) {
      coeffMap.insert({term.getExponent().getZExtValue(), term});
    }
    return coeffMap;
  }

  // Compose two polynomials this(other)
  Derived compose(const Derived& other) const {
    // This = a_0 + a_1x + ... + a_nx^n
    // Other = b_0 + b_1x + ... + b_mx^m

    // Note this could be faster by using an FFT-based algorithm: evaluate
    // the composite polynomial at the roots of unity, then apply an iFFT.
    // Since the only current use case for this is to rescale a polynomial
    // to a different domain (e.g. [-1, 1] -> [a, b]), the `other` polynomial
    // is always degree 1, so this should not be a bottleneck.

    // Using Horner's method:
    // initialize to a_n
    assert(!terms.empty());
    Monomial init = Monomial(terms.back());
    APInt zero = APInt(apintBitWidth, 0);
    init.setExponent(zero);
    Derived result = Derived::fromMonomials(init).value();

    DenseMap<int64_t, Monomial> coeffMap = getCoeffMap();

    // For each term a_i x^i in decreasing degree order, compute
    // result = result * b(x) + a_i
    for (int i = getDegree() - 1; i >= 0; --i) {
      result = result.naiveMul(other);
      if (coeffMap.contains(i)) {
        Monomial constCoeff(coeffMap[i]);
        constCoeff.setExponent(zero);
        result = result.add(Derived::fromMonomials({constCoeff}).value());
      }
    }

    result.canonicalize();
    return result;
  }

  Derived monomialMul(int exponent) const {
    SmallVector<Monomial> newTerms;
    for (auto& term : getTerms()) {
      Monomial newMonomial;
      newMonomial.setCoefficient(term.getCoefficient());
      newMonomial.setExponent(term.getExponent() + exponent);
      newTerms.push_back(newMonomial);
    }
    return Derived(newTerms);
  }

  Derived scale(CoefficientType scalar) const {
    SmallVector<Monomial> newTerms;
    for (auto& term : getTerms()) {
      newTerms.emplace_back(term.scale(scalar));
    }
    return Derived(newTerms);
  }

  virtual Derived sub(const Derived& other) const = 0;

  // Prints polynomial to 'os'.
  void print(raw_ostream& os) const { print(os, " + ", "**"); }

  void dump() const {
    std::string result;
    llvm::raw_string_ostream os(result);
    print(os);
    std::cout << os.str() << "\n";
  }

  // Prints polynomial so that it can be used as a valid identifier
  std::string toIdentifier() const {
    std::string result;
    llvm::raw_string_ostream os(result);
    print(os, "_", "");
    return os.str();
  }

  // Returns a zero polynomial
  static Derived zero() {
    SmallVector<Monomial> monomials;
    return Derived(monomials);
  }

  bool isZero() const { return getTerms().empty(); }

  unsigned getDegree() const {
    return terms.back().getExponent().getZExtValue();
  }

  ArrayRef<Monomial> getTerms() const { return terms; }

  template <class D, typename T, typename C>
  friend ::llvm::hash_code hash_value(const PolynomialBase<D, T, C>& arg);

 private:
  // The monomial terms for this polynomial.
  SmallVector<Monomial> terms;
};

/// A single-variable polynomial with integer coefficients.
///
/// Eg: x^1024 + x + 1
class IntPolynomial final
    : public PolynomialBase<IntPolynomial, IntMonomial, APInt> {
 public:
  explicit IntPolynomial(ArrayRef<IntMonomial> terms) : PolynomialBase(terms) {}

  // Returns a Polynomial from a list of monomials.
  // Fails if two monomials have the same exponent.
  static FailureOr<IntPolynomial> fromMonomials(
      ArrayRef<IntMonomial> monomials);

  /// Returns a polynomial with coefficients given by `coeffs`. The value
  /// coeffs[i] is converted to a monomial with exponent i.
  static IntPolynomial fromCoefficients(ArrayRef<int64_t> coeffs);

  IntPolynomial sub(const IntPolynomial& other) const override {
    return add(other.scale(APInt(apintBitWidth, -1)));
  }
};

/// A single-variable polynomial with double coefficients.
///
/// Eg: 1.0 x^1024 + 3.5 x + 1e-05
class FloatPolynomial final
    : public PolynomialBase<FloatPolynomial, FloatMonomial, APFloat> {
 public:
  explicit FloatPolynomial(ArrayRef<FloatMonomial> terms)
      : PolynomialBase(terms) {}

  // Returns a Polynomial from a list of monomials.
  // Fails if two monomials have the same exponent.
  static FailureOr<FloatPolynomial> fromMonomials(
      ArrayRef<FloatMonomial> monomials);

  /// Returns a polynomial with coefficients given by `coeffs`. The value
  /// coeffs[i] is converted to a monomial with exponent i.
  static FloatPolynomial fromCoefficients(ArrayRef<double> coeffs);

  FloatPolynomial sub(const FloatPolynomial& other) const override {
    return add(other.scale(APFloat(-1.)));
  }
};

// Make Polynomials hashable.
template <class D, typename T, typename C>
inline ::llvm::hash_code hash_value(const PolynomialBase<D, T, C>& arg) {
  return ::llvm::hash_combine_range(arg.terms.begin(), arg.terms.end());
}

template <class D, typename T>
inline ::llvm::hash_code hash_value(const MonomialBase<D, T>& arg) {
  return llvm::hash_combine(::llvm::hash_value(arg.coefficient),
                            ::llvm::hash_value(arg.exponent));
}

template <class D, typename T, typename C>
inline raw_ostream& operator<<(raw_ostream& os,
                               const PolynomialBase<D, T, C>& polynomial) {
  polynomial.print(os);
  return os;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_POLYNOMIAL_H_
