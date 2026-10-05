#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
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

// Evaluates the initial approximation y_0 ~= 1/x as an ArithmeticDag, using
// Paterson-Stockmeyer on the input rescaled from [lower, upper] to [-1, 1].
NodePtr initialApproximationDag(double x, double lower, double upper,
                                int64_t degree) {
  ChebyshevPolynomial poly =
      goldschmidtInitialApproximation(lower, upper, degree);
  std::vector<double> coefficients;
  for (const auto& term : poly.getTerms()) {
    coefficients.push_back(term.convertToDouble());
  }
  LiteralDouble scaledX = (2 * x - (lower + upper)) / (upper - lower);
  return patersonStockmeyerChebyshevPolynomialEvaluation(
      scaledX, coefficients, kMinCoeffs, kernel::DagType::floatTy(64));
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

TEST(GoldschmidtLoopTest, SplitLevels) {
  GoldschmidtLevelSplit split = splitGoldschmidtLevels(3);
  EXPECT_EQ(split.chebyshevDegree, 1);
  EXPECT_EQ(split.numIterations, 1);

  split = splitGoldschmidtLevels(7);
  EXPECT_EQ(split.chebyshevDegree, 7);
  EXPECT_EQ(split.numIterations, 3);

  split = splitGoldschmidtLevels(8);
  EXPECT_EQ(split.chebyshevDegree, 15);
  EXPECT_EQ(split.numIterations, 3);
}

TEST(GoldschmidtLoopTest, ZeroIterationsReturnsInitialApproximation) {
  auto x = NodeTy::leaf(LiteralDouble(0.5));
  auto y0 = NodeTy::constantScalar(1.9, kernel::DagType::floatTy(64));
  EXPECT_EQ(
      goldschmidtLoop<LiteralDouble>(x, y0, 0, kernel::DagType::floatTy(64)),
      y0);
}

TEST(GoldschmidtLoopTest, MatchesUnrolledReference) {
  for (int64_t numIterations = 1; numIterations <= 4; ++numIterations) {
    for (double x : {0.3, 0.5, 1.0, 1.7}) {
      double y0 = 1.0 / x + 0.1;  // A deliberately poor initial guess.
      auto result = goldschmidtLoop<LiteralDouble>(
          NodeTy::leaf(LiteralDouble(x)), NodeTy::leaf(LiteralDouble(y0)),
          numIterations, kernel::DagType::floatTy(64));
      EXPECT_NEAR(evaluate(result), goldschmidtReference(x, y0, numIterations),
                  1e-12)
          << "x=" << x << ", numIterations=" << numIterations;
    }
  }
}

TEST(GoldschmidtLoopTest, ApproximatesInverseOnDefaultDomain) {
  double lower = 0.1;
  double upper = 2.0;
  GoldschmidtLevelSplit split = splitGoldschmidtLevels(7);

  for (double x = lower; x <= upper; x += 0.05) {
    NodePtr y0 =
        initialApproximationDag(x, lower, upper, split.chebyshevDegree);
    double initialError = std::abs(x * evaluate(y0) - 1);

    auto result = goldschmidtLoop<LiteralDouble>(NodeTy::leaf(LiteralDouble(x)),
                                                 y0, split.numIterations,
                                                 kernel::DagType::floatTy(64));
    double finalError = std::abs(x * evaluate(result) - 1);

    // x * y_n = 1 - e_0^(2^n), so the relative error is squared on each
    // iteration.
    EXPECT_NEAR(finalError,
                std::pow(initialError, std::pow(2, split.numIterations)), 1e-12)
        << "x=" << x;
    EXPECT_LE(finalError, initialError) << "x=" << x;
  }
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
