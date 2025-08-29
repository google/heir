#include <cmath>
#include <map>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Polynomial/PatersonStockmeyer.h"
#include "lib/Utils/Polynomial/PolynomialTestVisitors.h"

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

using kernel::ArithmeticDagNode;
using kernel::LiteralDouble;

// Helper function to evaluate a polynomial using Paterson-Stockmeyer method
double evalPatersonStockmeyerPolynomial(
    double x, const std::map<int64_t, double>& coefficients) {
  auto x_node = ArithmeticDagNode<LiteralDouble>::leaf(LiteralDouble(x));
  auto result_node =
      patersonStockmeyerMonomialPolynomialEvaluation(x_node, coefficients);

  test::EvalVisitor visitor;
  return result_node->visit(visitor);
}

// Helper function to compute multiplicative depth
int computeMultiplicativeDepth(double x,
                               const std::map<int64_t, double>& coefficients) {
  auto x_node = ArithmeticDagNode<LiteralDouble>::leaf(LiteralDouble(x));
  auto result_node =
      patersonStockmeyerMonomialPolynomialEvaluation(x_node, coefficients);

  test::MultiplicativeDepthVisitor visitor;
  return static_cast<int>(result_node->visit(visitor));
}

TEST(PatersonStockmeyerEvaluationTest, ConstantPolynomial) {
  std::map<int64_t, double> coefficients = {{0, 5.0}};

  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(2.0, coefficients), 5.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(0.0, coefficients), 5.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(-3.5, coefficients), 5.0, 1e-10);
}

TEST(PatersonStockmeyerEvaluationTest, LinearPolynomial) {
  // p(x) = 3 + 2x
  std::map<int64_t, double> coefficients = {{0, 3.0}, {1, 2.0}};

  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(0.0, coefficients), 3.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(1.0, coefficients), 5.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(2.0, coefficients), 7.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(-1.0, coefficients), 1.0, 1e-10);
}

TEST(PatersonStockmeyerEvaluationTest, QuadraticPolynomial) {
  // p(x) = 1 + 2x + 3x^2
  std::map<int64_t, double> coefficients = {{0, 1.0}, {1, 2.0}, {2, 3.0}};

  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(0.0, coefficients), 1.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(1.0, coefficients), 6.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(2.0, coefficients), 17.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(-1.0, coefficients), 2.0, 1e-10);
}

TEST(PatersonStockmeyerEvaluationTest, HigherDegreePolynomial) {
  // p(x) = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6
  std::map<int64_t, double> coefficients = {
      {0, 1.0}, {1, 1.0}, {2, 1.0}, {3, 1.0}, {4, 1.0}, {5, 1.0}, {6, 1.0}};

  // For x=2: 1 + 2 + 4 + 8 + 16 + 32 + 64 = 127
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(2.0, coefficients), 127.0,
              1e-10);

  // For x=0: should be 1
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(0.0, coefficients), 1.0, 1e-10);

  // For x=1: should be 7
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(1.0, coefficients), 7.0, 1e-10);
}

TEST(PatersonStockmeyerEvaluationTest, SparsePolynomial) {
  // p(x) = 2 + 5x^3 + x^7 (missing x, x^2, x^4, x^5, x^6 terms)
  std::map<int64_t, double> coefficients = {{0, 2.0}, {3, 5.0}, {7, 1.0}};

  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(0.0, coefficients), 2.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(1.0, coefficients), 8.0, 1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(2.0, coefficients), 170.0,
              1e-10);  // 2 + 5*8 + 1*128
}

TEST(PatersonStockmeyerEvaluationTest, EmptyPolynomial) {
  std::map<int64_t, double> coefficients = {};

  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(5.0, coefficients), 0.0, 1e-10);
}

TEST(PatersonStockmeyerEvaluationTest, MultiplicativeDepthAdvantage) {
  std::map<int64_t, double> coefficients;
  for (int i = 0; i <= 15; ++i) {
    coefficients[i] = 1.0;
  }

  int depth = computeMultiplicativeDepth(2.0, coefficients);
  EXPECT_EQ(depth, 5);
}

TEST(PatersonStockmeyerEvaluationTest, CompareWithDirectEvaluation) {
  // Test against direct polynomial evaluation for verification
  // p(x) = 2 + 3x^2 + x^4
  std::map<int64_t, double> coefficients = {{0, 2.0}, {2, 3.0}, {4, 1.0}};
  double x = 1.5;
  double expected = 2.0 + 3.0 * x * x + 1.0 * x * x * x * x;
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(x, coefficients), expected,
              1e-10);
}

TEST(PatersonStockmeyerEvaluationTest, DegreeOneOptimization) {
  std::map<int64_t, double> coefficients = {{0, 3.0}, {1, 2.0}};
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(5.0, coefficients), 13.0, 1e-10);
}

TEST(PatersonStockmeyerEvaluationTest, PowerPrecomputationVerification) {
  // p(x) = x^8
  std::map<int64_t, double> coefficients = {{8, 1.0}};
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(2.0, coefficients), 256.0,
              1e-10);
  EXPECT_NEAR(evalPatersonStockmeyerPolynomial(3.0, coefficients), 6561.0,
              1e-10);
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
