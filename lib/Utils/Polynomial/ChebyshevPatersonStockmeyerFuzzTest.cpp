#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Polynomial/PolynomialTestVisitors.h"

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

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

// Paterson-Stockmeyer evaluation of a Chebyshev polynomial
double psEvalChebyshevPolynomial(const std::vector<double>& coefficients,
                                 double x) {
  if (coefficients.empty()) return 0.0;

  kernel::LiteralDouble xNode = x;
  auto resultNode =
      patersonStockmeyerChebyshevPolynomialEvaluation(xNode, coefficients);

  if (!resultNode) return 0.0;

  test::EvalVisitor visitor;
  return resultNode->visit(visitor);
}

void patersonStockmeyerMatchesNaive(const std::vector<double>& coefficients,
                                    double x) {
  assert(x >= -1.0 && x <= 1.0);
  double expected = naiveEvalChebyshevPolynomial(coefficients, x);
  double actual = psEvalChebyshevPolynomial(coefficients, x);
  double tolerance = 1e-15;
  EXPECT_NEAR(expected, actual, tolerance);
}

// Fuzz test for Paterson-Stockmeyer evaluation
FUZZ_TEST(ChebyshevPatersonStockmeyerFuzzTest, patersonStockmeyerMatchesNaive)
    .WithDomains(
        /*coefficients=*/fuzztest::VectorOf(fuzztest::InRange(-100.0, 100.0))
            .WithMinSize(1)
            .WithMaxSize(50),
        /*x=*/fuzztest::InRange(-1.0, 1.0));

// Throw in some specific cases.
TEST(ChebyshevPatersonStockmeyerFuzzTest, SingleCoefficient) {
  patersonStockmeyerMatchesNaive({5.0}, 0.5);
}

TEST(ChebyshevPatersonStockmeyerFuzzTest, TwoCoefficients) {
  patersonStockmeyerMatchesNaive({1.0, 2.5}, 0.5);
}

TEST(ChebyshevPatersonStockmeyerFuzzTest, MultipleCoefficients) {
  patersonStockmeyerMatchesNaive({3.0, -2.0, 0.5, 1.0, -0.5}, 0.7);
}

TEST(ChebyshevPatersonStockmeyerFuzzTest, ZeroCoefficients) {
  patersonStockmeyerMatchesNaive({0.0, 0.0, 1.0, 0.0}, 0.3);
}

TEST(ChebyshevPatersonStockmeyerFuzzTest, BoundaryValues) {
  patersonStockmeyerMatchesNaive({1.0, 2.0, 3.0}, -1.0);
  patersonStockmeyerMatchesNaive({1.0, 2.0, 3.0}, 1.0);
  patersonStockmeyerMatchesNaive({1.0, 2.0, 3.0}, 0.0);
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
