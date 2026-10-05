#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "fuzztest/fuzztest.h"  // from @fuzztest
#include "gtest/gtest.h"        // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Polynomial/GoldschmidtLoop.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Polynomial/PolynomialTestVisitors.h"

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

using kernel::LiteralDouble;
using NodeTy = kernel::ArithmeticDagNode<LiteralDouble>;
using NodePtr = std::shared_ptr<NodeTy>;

double evaluate(const NodePtr& node) {
  test::EvalVisitor visitor;
  return visitor.process(node)[0];
}

double goldschmidtDagEval(double x, double y0, int64_t numIterations) {
  return evaluate(goldschmidtLoop<LiteralDouble>(
      NodeTy::leaf(LiteralDouble(x)), NodeTy::leaf(LiteralDouble(y0)),
      numIterations, kernel::DagType::floatTy(64)));
}

// Reference implementation of the unrolled Goldschmidt iteration.
double goldschmidtReference(double x, double y0, int64_t numIterations) {
  double y = y0;
  double e = 1 - x * y0;
  for (int64_t i = 0; i < numIterations; ++i) {
    y = y * (1 + e);
    e = e * e;
  }
  return y;
}

// The loop DAG (with its last iteration peeled) computes the same value as the
// straightforward unrolled iteration. The initial guess is y_0 = (1 - e_0) / x
// for a fuzzed initial relative error e_0.
void loopMatchesUnrolledReference(double x, double e0, int64_t numIterations) {
  double y0 = (1 - e0) / x;
  double expected = goldschmidtReference(x, y0, numIterations);
  double actual = goldschmidtDagEval(x, y0, numIterations);
  EXPECT_NEAR(actual, expected, 1e-12 * std::max(1.0, std::abs(expected)));
}

FUZZ_TEST(GoldschmidtLoopFuzzTest, loopMatchesUnrolledReference)
    .WithDomains(fuzztest::InRange(1e-3, 1e3), fuzztest::InRange(-0.99, 0.99),
                 fuzztest::InRange<int64_t>(0, 6));

// relative error squares on every iteration.
void relativeErrorSquaresEachIteration(double x, double e0,
                                       int64_t numIterations) {
  double y0 = (1 - e0) / x;
  double finalError = x * goldschmidtDagEval(x, y0, numIterations) - 1;
  double expectedError = -std::pow(e0, std::pow(2, numIterations));
  EXPECT_NEAR(finalError, expectedError, 1e-12);
  EXPECT_LE(std::abs(finalError), std::abs(e0) + 1e-12);
}

FUZZ_TEST(GoldschmidtLoopFuzzTest, relativeErrorSquaresEachIteration)
    .WithDomains(/*x=*/fuzztest::InRange(1e-3, 1e3),
                 /*e0=*/fuzztest::InRange(-0.99, 0.99),
                 /*numIterations=*/fuzztest::InRange<int64_t>(0, 6));

// End to end: build y_0 from the Chebyshev initial approximation evaluated with
// Paterson-Stockmeyer, run the Goldschmidt loop, and check the result against
// 1/x.
void approximatesInverse(double lower, double ratio, int64_t levels, double t) {
  double upper = lower * ratio;
  double x = lower + t * (upper - lower);
  GoldschmidtLevelSplit split = splitGoldschmidtLevels(levels);

  ChebyshevPolynomial poly =
      goldschmidtInitialApproximation(lower, upper, split.chebyshevDegree);
  std::vector<double> coefficients;
  for (const auto& term : poly.getTerms()) {
    coefficients.push_back(term.convertToDouble());
  }
  LiteralDouble scaledX = (2 * x - (lower + upper)) / (upper - lower);
  NodePtr y0 = patersonStockmeyerChebyshevPolynomialEvaluation(
      scaledX, coefficients, kMinCoeffs, kernel::DagType::floatTy(64));
  ASSERT_NE(y0, nullptr);

  double initialError = 1 - x * evaluate(y0);
  double result = evaluate(goldschmidtLoop<LiteralDouble>(
      NodeTy::leaf(LiteralDouble(x)), y0, split.numIterations,
      kernel::DagType::floatTy(64)));
  double finalError = 1 - x * result;

  // Goldschmidt only converges when the initial guess is within a factor of 2
  // of 1/x. Outside of that, the iteration amplifies the error, which is still
  // consistent with the closed form below.
  double expectedError =
      std::pow(initialError, std::pow(2, split.numIterations));
  EXPECT_NEAR(finalError, expectedError,
              1e-9 * std::max(1.0, std::abs(expectedError)))
      << "x=" << x << ", domain=[" << lower << ", " << upper
      << "], levels=" << levels;
  if (std::abs(initialError) < 1) {
    EXPECT_LE(std::abs(finalError), std::abs(initialError) + 1e-12)
        << "x=" << x << ", domain=[" << lower << ", " << upper
        << "], levels=" << levels;
  }
}

FUZZ_TEST(GoldschmidtLoopFuzzTest, approximatesInverse)
    .WithDomains(fuzztest::InRange(1e-2, 10.0), fuzztest::InRange(1.01, 100.0),
                 fuzztest::InRange<int64_t>(3, 9), fuzztest::InRange(0.0, 1.0));

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
