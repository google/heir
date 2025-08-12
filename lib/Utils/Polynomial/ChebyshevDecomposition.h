#ifndef LIB_UTILS_POLYNOMIAL_CHEBYSHEVDECOMPOSITION_H_
#define LIB_UTILS_POLYNOMIAL_CHEBYSHEVDECOMPOSITION_H_

#include <vector>

namespace mlir {
namespace heir {
namespace polynomial {

// Represents the polynomial in the Chebyshev polynomials basis. Namely,
// let p is ChebyshevBasisPolynomial, then it represents the polynomial:
// P(x) = p[0]*T_0(x) + p[1]T_1(x) + ... + p[k] T_k(x).
using ChebyshevBasisPolynomial = std::vector<double>;

// Represents the polynomial:
// p = coeffs[0] + coeffs[1]*T_k + coeffs[2]*T_k^2 + ... + coeffs[l]*T_k^l
// where, k = generatorDegree, T_k is k-th Chebyshev polynomial and qs are
// polynomials in the basis of the Chebyshev polynomials.
struct ChebyshevDecomposition {
  int generatorDegree;
  std::vector<ChebyshevBasisPolynomial> coeffs;
};

// Decomposes the polynomial in the Chebyshev polynomials basis.
ChebyshevDecomposition decompose(
    const ChebyshevBasisPolynomial& cheb_polynomial, int decomposition_degree);

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_CHEBYSHEVDECOMPOSITION_H_
