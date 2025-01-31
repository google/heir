
#include <cmath>
#include <cstdint>

#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace approximation {

using ::llvm::APFloat;
using ::llvm::SmallVector;
using ::mlir::heir::polynomial::FloatPolynomial;

// When we move to C++20, we can use std::numbers::pi
inline constexpr double kPi = 3.14159265358979323846;

void getChebyshevPoints(int64_t numPoints, SmallVector<APFloat> &results) {
  if (numPoints == 0) {
    return;
  }
  if (numPoints == 1) {
    results.push_back(APFloat(0.));
    return;
  }

  // The values are most simply described as
  //
  //     cos(pi * j / (n-1)) for 0 <= j <= n-1.
  //
  // But to enforce symmetry around the origin---broken by slight numerical
  // inaccuracies---and the left-to-right ordering, we apply the identity
  //
  //     cos(x + pi) = -cos(x) = sin(x - pi/2)
  //
  // to arrive at
  //
  //     sin(pi*j/(n-1) - pi/2) = sin(pi * (2j - (n-1)) / (2(n-1)))
  //
  // An this is equivalent to the formula below, where the range of j is shifted
  // and rescaled from {0, ..., n-1} to {-n+1, -n+3, ..., n-3, n-1}.
  int64_t m = numPoints - 1;
  for (int64_t j = -m; j < m + 1; j += 2) {
    results.push_back(APFloat(std::sin(kPi * j / (2 * m))));
  }
}

void getChebyshevPolynomials(int64_t numPolynomials,
                             SmallVector<FloatPolynomial> &results) {
  if (numPolynomials < 1) return;

  if (numPolynomials >= 1) {
    // 1
    results.push_back(FloatPolynomial::fromCoefficients({1.}));
  }
  if (numPolynomials >= 2) {
    // 2x
    results.push_back(FloatPolynomial::fromCoefficients({0., 2.}));
  }

  if (numPolynomials <= 2) return;

  for (int64_t i = 2; i < numPolynomials; ++i) {
    auto &last = results.back();
    auto &secondLast = results[results.size() - 2];
    results.push_back(last.monomialMul(1).scale(APFloat(2.)).sub(secondLast));
  }
}

}  // namespace approximation
}  // namespace heir
}  // namespace mlir
