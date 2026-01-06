#ifndef LIB_UTILS_POLYNOMIAL_CHEBYSHEVDECOMPOSITIONTESTUTILS_H_
#define LIB_UTILS_POLYNOMIAL_CHEBYSHEVDECOMPOSITIONTESTUTILS_H_

#include <vector>

namespace mlir {
namespace heir {
namespace polynomial {

// Naive evaluation of a Chebyshev polynomial using direct computation
// of each Chebyshev basis polynomial.
double naiveEvalChebyshevPolynomial(const std::vector<double>& coefficients,
                                    double x);

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_POLYNOMIAL_CHEBYSHEVDECOMPOSITIONTESTUTILS_H_
