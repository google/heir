#include <cmath>
#include <map>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Utils/Polynomial/Horner.h"
#include "lib/Utils/Polynomial/PolynomialTestVisitors.h"

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

using kernel::ArithmeticDagNode;

// Helper function to evaluate a polynomial using Horner's method
double evalHornerPolynomial(double x,
                            const std::map<int64_t, double>& coefficients,
                            double minCoeffThreshold = 1e-12) {
  auto x_node = ArithmeticDagNode<double>::leaf(x);
  auto result_node = hornerMonomialPolynomialEvaluation(x_node, coefficients,
                                                        minCoeffThreshold);

  test::EvalVisitor visitor;
  return result_node->visit(visitor);
}

// Helper function to compute multiplicative depth
int computeMultiplicativeDepth(double x,
                               const std::map<int64_t, double>& coefficients,
                               double minCoeffThreshold = 1e-12) {
  auto x_node = ArithmeticDagNode<double>::leaf(x);
  auto result_node = hornerMonomialPolynomialEvaluation(x_node, coefficients,
                                                        minCoeffThreshold);

  test::MultiplicativeDepthVisitor visitor;
  return static_cast<int>(result_node->visit(visitor));
}

TEST(HornerEvaluationTest, ConstantPolynomial) {
  std::map<int64_t, double> coefficients = {{0, 5.0}};

  EXPECT_NEAR(evalHornerPolynomial(2.0, coefficients), 5.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(0.0, coefficients), 5.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(-3.5, coefficients), 5.0, 1e-10);
}

TEST(HornerEvaluationTest, LinearPolynomial) {
  // p(x) = 3 + 2x
  std::map<int64_t, double> coefficients = {{0, 3.0}, {1, 2.0}};

  EXPECT_NEAR(evalHornerPolynomial(0.0, coefficients), 3.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(1.0, coefficients), 5.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(2.0, coefficients), 7.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(-1.0, coefficients), 1.0, 1e-10);
}

TEST(HornerEvaluationTest, QuadraticPolynomial) {
  // p(x) = 1 + 2x + 3x^2
  std::map<int64_t, double> coefficients = {{0, 1.0}, {1, 2.0}, {2, 3.0}};

  EXPECT_NEAR(evalHornerPolynomial(0.0, coefficients), 1.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(1.0, coefficients), 6.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(2.0, coefficients), 17.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(-1.0, coefficients), 2.0, 1e-10);
}

TEST(HornerEvaluationTest, SparsePolynomial) {
  // p(x) = 2 + 5x^3 + x^7 (missing x, x^2, x^4, x^5, x^6 terms)
  std::map<int64_t, double> coefficients = {{0, 2.0}, {3, 5.0}, {7, 1.0}};

  EXPECT_NEAR(evalHornerPolynomial(0.0, coefficients), 2.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(1.0, coefficients), 8.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(2.0, coefficients), 170.0,
              1e-10);  // 2 + 5*8 + 1*128
}

TEST(HornerEvaluationTest, EmptyPolynomial) {
  std::map<int64_t, double> coefficients = {};

  EXPECT_NEAR(evalHornerPolynomial(5.0, coefficients), 0.0, 1e-10);
}

TEST(HornerEvaluationTest, SmallCoefficientFiltering) {
  // p(x) = 1 + 1e-15*x + 2x^2 + 1e-16*x^3
  std::map<int64_t, double> coefficients = {
      {0, 1.0}, {1, 1e-15}, {2, 2.0}, {3, 1e-16}};

  // With default threshold (1e-12), small coefficients should be dropped
  // Result should be 1 + 2x^2
  EXPECT_NEAR(evalHornerPolynomial(0.0, coefficients), 1.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(1.0, coefficients), 3.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(2.0, coefficients), 9.0, 1e-10);  // 1 + 2*4

  // With stricter threshold (1e-17), more coefficients should be kept
  // Result should be 1 + 1e-15*x + 2x^2 + 1e-16*x^3
  EXPECT_NEAR(evalHornerPolynomial(1.0, coefficients, 1e-17),
              3.0 + 1e-15 + 1e-16, 1e-12);
}

TEST(HornerEvaluationTest, AllSmallCoefficients) {
  // All coefficients below threshold
  std::map<int64_t, double> coefficients = {{0, 1e-15}, {1, 1e-14}, {2, 1e-13}};

  // Should return 0 when all coefficients are filtered out
  EXPECT_NEAR(evalHornerPolynomial(5.0, coefficients), 0.0, 1e-10);
}

TEST(HornerEvaluationTest, MultiplicativeDepthLinear) {
  // Linear polynomial: a0 + a1*x
  std::map<int64_t, double> coefficients = {{0, 3.0}, {1, 2.0}};

  // Horner for linear: (a1)*x + a0 has depth 0 (no mults between non-constants)
  EXPECT_EQ(computeMultiplicativeDepth(2.0, coefficients), 0);
}

TEST(HornerEvaluationTest, MultiplicativeDepthQuadratic) {
  // Quadratic polynomial: a0 + a1*x + a2*x^2
  std::map<int64_t, double> coefficients = {{0, 1.0}, {1, 2.0}, {2, 3.0}};

  // Horner for quadratic: ((a2)*x + a1)*x + a0 has depth 1 (one x
  // multiplication cascade)
  EXPECT_EQ(computeMultiplicativeDepth(2.0, coefficients), 1);
}

TEST(HornerEvaluationTest, MultiplicativeDepthHighDegree) {
  // Higher degree polynomial
  std::map<int64_t, double> coefficients = {
      {0, 1.0}, {1, 2.0}, {2, 3.0}, {3, 4.0}, {4, 5.0}};

  // Horner for degree 4: depth should be 3 (cascade of mults)
  EXPECT_EQ(computeMultiplicativeDepth(2.0, coefficients), 3);
}

TEST(HornerEvaluationTest, NumericalAccuracy) {
  // Test with a polynomial known to have numerical issues: Wilkinson's
  // polynomial (simplified) p(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
  std::map<int64_t, double> coefficients = {
      {0, -6.0}, {1, 11.0}, {2, -6.0}, {3, 1.0}};

  // Check roots
  EXPECT_NEAR(evalHornerPolynomial(1.0, coefficients), 0.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(2.0, coefficients), 0.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(3.0, coefficients), 0.0, 1e-10);

  // Check other values
  EXPECT_NEAR(evalHornerPolynomial(0.0, coefficients), -6.0, 1e-10);
  EXPECT_NEAR(evalHornerPolynomial(4.0, coefficients), 6.0,
              1e-10);  // (4-1)(4-2)(4-3) = 3*2*1
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
