#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyerTestUtils.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace heir {
namespace polynomial {

namespace {

// Compute Chebyshev polynomial T_n(x) using the standard recurrence relation.
// T_0(x) = 1
// T_1(x) = x
// T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
double computeChebyshevPolynomial(int64_t n, double x) {
  if (n == 0) return 1.0;
  if (n == 1) return x;

  double tPrev = 1.0;  // T_0(x)
  double tCurr = x;    // T_1(x)

  for (int64_t i = 2; i <= n; i++) {
    double tNext = 2.0 * x * tCurr - tPrev;
    tPrev = tCurr;
    tCurr = tNext;
  }

  return tCurr;
}

}  // namespace

// Naive evaluation of a Chebyshev polynomial using direct computation
// of each Chebyshev basis polynomial.
double naiveEvalChebyshevPolynomial(const std::vector<double>& coefficients,
                                    double x) {
  if (coefficients.empty()) return 0.0;

  double result = 0.0;
  for (size_t i = 0; i < coefficients.size(); i++) {
    result += coefficients[i] * computeChebyshevPolynomial(i, x);
  }
  return result;
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
