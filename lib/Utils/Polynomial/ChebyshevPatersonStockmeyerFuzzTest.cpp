#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyerTestUtils.h"
#include "lib/Utils/Polynomial/PolynomialTestVisitors.h"

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

// Paterson-Stockmeyer evaluation of a Chebyshev polynomial
double psEvalChebyshevPolynomial(const std::vector<double>& coefficients,
                                 double x) {
  if (coefficients.empty()) return 0.0;

  kernel::LiteralDouble xNode = x;
  // Use f64 type for double precision tests, default minCoeffThreshold
  auto resultNode =
      patersonStockmeyerChebyshevPolynomialEvaluation(
          xNode, coefficients, kMinCoeffs, kernel::DagType::floatTy(64));

  if (!resultNode) return 0.0;

  test::EvalVisitor visitor;
  return resultNode->visit(visitor);
}

void patersonStockmeyerMatchesNaive(const std::vector<double>& coefficients,
                                    double x) {
  assert(x >= -1.0 && x <= 1.0);
  double expected = naiveEvalChebyshevPolynomial(coefficients, x);
  double actual = psEvalChebyshevPolynomial(coefficients, x);
  double tolerance = 1e-11;
  EXPECT_NEAR(expected, actual, tolerance);
}

// Fuzz test for Paterson-Stockmeyer evaluation
FUZZ_TEST(ChebyshevPatersonStockmeyerFuzzTest, patersonStockmeyerMatchesNaive)
    .WithDomains(
        /*coefficients=*/fuzztest::VectorOf(fuzztest::InRange(-5.0, 5.0))
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
