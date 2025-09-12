#include "lib/Utils/Polynomial/Polynomial.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

namespace {

FloatPolynomial chebyshevToMonomial(const SmallVector<APFloat>& coefficients) {
  SmallVector<FloatPolynomial> chebPolys;
  chebPolys.reserve(coefficients.size());
  ChebyshevPolynomial::getChebyshevPolynomials(coefficients.size(), chebPolys);

  FloatPolynomial result = FloatPolynomial::zero();
  for (int64_t i = 0; i < coefficients.size(); ++i) {
    result = result.add(chebPolys[i].scale(coefficients[i]));
  }

  return result;
}

}  // namespace

template <typename PolyT, typename MonomialT>
FailureOr<PolyT> fromMonomialsImpl(ArrayRef<MonomialT> monomials) {
  // A polynomial's terms are canonically stored in order of increasing degree.
  auto monomialsCopy = llvm::SmallVector<MonomialT>(monomials);
  std::sort(monomialsCopy.begin(), monomialsCopy.end());

  // Ensure non-unique exponents are not present. Since we sorted the list by
  // exponent, a linear scan of adjacent monomials suffices.
  if (std::adjacent_find(monomialsCopy.begin(), monomialsCopy.end(),
                         [](const MonomialT& lhs, const MonomialT& rhs) {
                           return lhs.getExponent() == rhs.getExponent();
                         }) != monomialsCopy.end()) {
    return failure();
  }

  return PolyT(monomialsCopy);
}

FailureOr<IntPolynomial> IntPolynomial::fromMonomials(
    ArrayRef<IntMonomial> monomials) {
  return fromMonomialsImpl<IntPolynomial, IntMonomial>(monomials);
}

FailureOr<FloatPolynomial> FloatPolynomial::fromMonomials(
    ArrayRef<FloatMonomial> monomials) {
  return fromMonomialsImpl<FloatPolynomial, FloatMonomial>(monomials);
}

template <typename PolyT, typename MonomialT, typename CoeffT>
PolyT fromCoefficientsImpl(ArrayRef<CoeffT> coeffs) {
  llvm::SmallVector<MonomialT> monomials;
  auto size = coeffs.size();
  monomials.reserve(size);
  for (size_t i = 0; i < size; i++) {
    if (coeffs[i] != 0) monomials.emplace_back(coeffs[i], i);
  }
  auto result = PolyT::fromMonomials(monomials);
  // Construction guarantees unique exponents, so the failure mode of
  // fromMonomials can be bypassed.
  assert(succeeded(result));
  return result.value();
}

IntPolynomial IntPolynomial::fromCoefficients(ArrayRef<int64_t> coeffs) {
  return fromCoefficientsImpl<IntPolynomial, IntMonomial, int64_t>(coeffs);
}

FloatPolynomial FloatPolynomial::fromCoefficients(ArrayRef<double> coeffs) {
  return fromCoefficientsImpl<FloatPolynomial, FloatMonomial, double>(coeffs);
}

FloatPolynomial ChebyshevPolynomial::toStandardBasis() const {
  return chebyshevToMonomial(terms);
}

ArrayAttr ChebyshevPolynomial::getCoefficientsArrayAttr(
    mlir::MLIRContext* context) const {
  SmallVector<Attribute> coeffs;
  coeffs.reserve(terms.size());
  for (const auto& coeff : terms) {
    coeffs.push_back(FloatAttr::get(mlir::Float64Type::get(context), coeff));
  }
  return ArrayAttr::get(context, coeffs);
}

ChebyshevPolynomial ChebyshevPolynomial::compose(
    const FloatPolynomial& other) const {
  assert(other.getDegree() == 1 &&
         "Only composition with linear polynomials is supported");
  // This implements the composition of a Chebyshev series with a linear
  // function H(y(x)) where y(x) = ax+b. The method is based on a recurrence
  // derived from Clenshaw's algorithm for evaluating a Chebyshev series and
  // specialized to Chebyshev polynomials of the first kind on the interval [-1,
  // 1]. The original paper handles shifted Chebyshev polynomials on [0, 1].
  //
  // Let H(x) = sum c_i T_i(x). We want to compute H(y).
  // We use the recurrence d_k = c_k + 2*y*d_{k+1} - d_{k+2}, with
  // d_{n+1}=d_{n+2}=0. The result is H(y) = (c_0 + d_0 - d_2)/2.
  // See
  // https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
  // for a reference.
  //
  // Since y is a polynomial in x, the d_k are also polynomials in x. We can
  // compute the Chebyshev coefficients of each d_k iteratively.

  int n = getDegree();
  if (n < 0) {
    return ChebyshevPolynomial::zero();
  }

  auto otherCoeffs = other.getCoeffMap();
  APFloat b = otherCoeffs.contains(0) ? otherCoeffs.at(0).getCoefficient()
                                      : APFloat(0.0);
  APFloat a = otherCoeffs.contains(1) ? otherCoeffs.at(1).getCoefficient()
                                      : APFloat(0.0);

  SmallVector<APFloat> c = terms;

  SmallVector<APFloat> d_kplus2;
  SmallVector<APFloat> d_kplus1 = {c[n]};
  SmallVector<APFloat> d_2_coeffs;

  for (int k = n - 1; k >= 0; --k) {
    int deg_kplus1 = n - (k + 1);
    // d_k has degree n-k.
    SmallVector<APFloat> d_k(n - k + 1, APFloat(0.0));

    // Compute coefficients of 2*y*d_{k+1} in the Chebyshev basis.
    if (deg_kplus1 >= 0) {
      // T_0 coeff of 2*y*d_{k+1}
      APFloat d_kplus1_1 = (deg_kplus1 >= 1) ? d_kplus1[1] : APFloat(0.0);
      d_k[0] = a * d_kplus1_1 + b * d_kplus1[0] * APFloat(2.0);

      // T_1 coeff of 2*y*d_{k+1}
      if (n - k >= 1) {
        APFloat d_kplus1_2 = (deg_kplus1 >= 2) ? d_kplus1[2] : APFloat(0.0);
        d_k[1] = a * (d_kplus1[0] * APFloat(2.0) + d_kplus1_2) +
                 b * d_kplus1_1 * APFloat(2.0);
      }

      // T_m coeffs for m >= 2
      for (int m = 2; m <= n - k; ++m) {
        APFloat d_kplus1_m_minus_1 =
            (m - 1 <= deg_kplus1) ? d_kplus1[m - 1] : APFloat(0.0);
        APFloat d_kplus1_m_plus_1 =
            (m + 1 <= deg_kplus1) ? d_kplus1[m + 1] : APFloat(0.0);
        APFloat d_kplus1_m = (m <= deg_kplus1) ? d_kplus1[m] : APFloat(0.0);
        d_k[m] = a * (d_kplus1_m_minus_1 + d_kplus1_m_plus_1) +
                 b * d_kplus1_m * APFloat(2.0);
      }
    }

    // Add c_k
    d_k[0] = d_k[0] + c[k];

    // Subtract d_{k+2}
    for (int m = 0; m < d_kplus2.size(); ++m) {
      d_k[m] = d_k[m] - d_kplus2[m];
    }

    if (k == 2) {
      d_2_coeffs = d_k;
    }
    d_kplus2 = d_kplus1;
    d_kplus1 = d_k;
  }

  // At this point, d_kplus1 holds the coefficients of d_0.
  // The final coefficients are (c0 + d_0 - d_2)/2.
  SmallVector<double> resultCoeffs(n + 1, 0.0);
  APFloat half(0.5);
  for (int i = 0; i <= n; ++i) {
    APFloat d_0_i = d_kplus1[i];
    APFloat d_2_i = (i < d_2_coeffs.size()) ? d_2_coeffs[i] : APFloat(0.0);
    resultCoeffs[i] = ((d_0_i - d_2_i) * half).convertToDouble();
  }

  // Add c0 / 2.
  resultCoeffs[0] += (c[0] * half).convertToDouble();

  return ChebyshevPolynomial(resultCoeffs);
}

void ChebyshevPolynomial::getChebyshevPolynomials(
    int64_t numPolynomials, SmallVector<FloatPolynomial>& results) {
  if (numPolynomials < 1) return;

  if (numPolynomials >= 1) {
    // 1
    results.push_back(FloatPolynomial::fromCoefficients({1.}));
  }
  if (numPolynomials >= 2) {
    // x
    results.push_back(FloatPolynomial::fromCoefficients({0., 1.}));
  }

  if (numPolynomials <= 2) return;

  for (int64_t i = 2; i < numPolynomials; ++i) {
    auto& last = results.back();
    auto& secondLast = results[results.size() - 2];
    results.push_back(last.monomialMul(1).scale(APFloat(2.)).sub(secondLast));
  }
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
