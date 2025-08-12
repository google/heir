#include "lib/Utils/Polynomial/ChebyshevDecomposition.h"

#include <cstdlib>
#include <utility>

namespace mlir {
namespace heir {
namespace polynomial {

namespace {

// Finds polynomials q, r in the Chebyshev basis, such that
// p = q*T_k + r.
std::pair<ChebyshevBasisPolynomial, ChebyshevBasisPolynomial> dividePolynomials(
    ChebyshevBasisPolynomial p, int k) {
  if (k >= p.size()) {
    return {{}, p};
  }
  ChebyshevBasisPolynomial q(p.size() - k, 0.0);
  for (int i = p.size() - 1; i >= k; --i) {
    if (p[i] == 0.0) {
      continue;
    }
    if (i == k) {
      q[0] = p[i];
    } else {
      // Formula: 2T_m(x)T_n(x) = T_{m+n}(x) + T_{|m-n|}(x) is used
      // https://en.wikipedia.org/wiki/Chebyshev_polynomials#Products_of_Chebyshev_polynomials
      // Namely
      // p_iT_i + ... = p_i(2T_kT_{i-k}-T_{|i-2k|}) + ... =
      // T_k*(2p_iT_{i-k}) +            // this term goes to q
      // -p_iT_{|i-2k|} + ...           // the rest stays in p
      // As a result on each iteration we decrease the degree of the polynomial
      // which we divide.
      q[i - k] = p[i] * 2.0;
      p[std::abs(i - 2 * k)] -= p[i];

      p[i] = 0.0;
    }
  }
  ChebyshevBasisPolynomial r(p.begin(), p.begin() + k);
  return {q, r};
}
}  // namespace

ChebyshevDecomposition decompose(
    const ChebyshevBasisPolynomial& cheb_polynomial, int decomposition_degree) {
  ChebyshevBasisPolynomial p = cheb_polynomial;
  ChebyshevDecomposition result{.generatorDegree = decomposition_degree};
  while (!p.empty()) {
    auto [next_p, q] = dividePolynomials(p, decomposition_degree);
    result.coeffs.push_back(q);
    p = next_p;
  }
  return result;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
