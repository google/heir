#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Approximation/CaratheodoryFejer.h"
#include "lib/Utils/Approximation/Taylor.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace approximation {
namespace {

struct ExpTaylorTestParams {
  int64_t k;
  double domainLower;
  double domainUpper;
  double step;
  double maxInfinityNormError;
};

class TaylorApproximationTest
    : public ::testing::TestWithParam<ExpTaylorTestParams> {};

TEST_P(TaylorApproximationTest, ExpAccuracyAcrossWholeDomain) {
  const ExpTaylorTestParams& params = GetParam();
  auto expFunc = [](const ::llvm::APFloat& x) {
    return ::llvm::APFloat(std::exp(x.convertToDouble()));
  };
  // Compare Taylor with k squarings against a Chebyshev polynomial of degree k.
  polynomial::FloatPolynomial chebyshevPoly =
      caratheodoryFejerApproximation(expFunc, params.k, params.domainLower,
                                     params.domainUpper)
          .toStandardBasis();

  double taylorInfinityNormError = 0.0;
  double taylorSumSqDiff = 0.0;
  double chebyshevSumSqDiff = 0.0;

  for (double x = params.domainLower; x <= params.domainUpper;
       x += params.step) {
    double expected = std::exp(x);
    double taylorActual = expTaylorApproximation(x, params.k);

    double chebyshevActual = 0.0;
    for (const auto& term : chebyshevPoly.getTerms()) {
      chebyshevActual += term.getCoefficient().convertToDouble() *
                         std::pow(x, term.getExponent().getZExtValue());
    }

    double taylorAbsError = std::abs(taylorActual - expected);
    double chebyshevAbsError = std::abs(chebyshevActual - expected);

    taylorInfinityNormError = std::max(taylorInfinityNormError, taylorAbsError);

    taylorSumSqDiff += taylorAbsError * taylorAbsError;
    chebyshevSumSqDiff += chebyshevAbsError * chebyshevAbsError;
  }

  // Verify infinity norm (max absolute error) for Taylor.
  EXPECT_LT(taylorInfinityNormError, params.maxInfinityNormError);

  // Compare L2 norm against Chebyshev of degree k.
  EXPECT_LT(taylorSumSqDiff, chebyshevSumSqDiff);
}

INSTANTIATE_TEST_SUITE_P(
    TaylorTests, TaylorApproximationTest,
    ::testing::Values(ExpTaylorTestParams{/*k=*/7, /*domainLower=*/-128.0,
                                          /*domainUpper=*/1.0, /*step=*/0.1,
                                          /*maxInfinityNormError=*/0.015},
                      ExpTaylorTestParams{/*k=*/14, /*domainLower=*/-16384.0,
                                          /*domainUpper=*/1.0, /*step=*/1.0,
                                          /*maxInfinityNormError=*/1e-4}));

}  // namespace
}  // namespace approximation
}  // namespace heir
}  // namespace mlir
