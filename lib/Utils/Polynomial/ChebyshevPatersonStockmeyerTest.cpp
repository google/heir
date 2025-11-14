#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Polynomial/PolynomialTestVisitors.h"

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

double evalChebyshevPolynomial(double x, std::vector<double> coefficients) {
  kernel::LiteralDouble x_node = x;
  auto result_node =
      patersonStockmeyerChebyshevPolynomialEvaluation(x_node, coefficients);

  test::EvalVisitor visitor;
  return result_node->visit(visitor);
}

int evalMultiplicativeDepth(double x, std::vector<double> coefficients) {
  kernel::LiteralDouble x_node = x;
  auto result_node =
      patersonStockmeyerChebyshevPolynomialEvaluation(x_node, coefficients);

  test::MultiplicativeDepthVisitor visitor;
  return static_cast<int>(result_node->visit(visitor));
}

TEST(PatersonStockmeyerChebyshevPolynomialEvaluation, ConstantPolynomial) {
  std::vector<double> coefficients = {5.0};  // Represents 5
  EXPECT_EQ(evalChebyshevPolynomial(0.5, coefficients), 5.0);
  EXPECT_EQ(evalMultiplicativeDepth(0.5, coefficients), -1);
}

TEST(PatersonStockmeyerChebyshevPolynomialEvaluation, LinearPolynomial) {
  std::vector<double> coefficients = {1.0, 2.5};  // Represents 1 + 2.5*T_1
  EXPECT_EQ(evalChebyshevPolynomial(0.0, coefficients), 1);
  EXPECT_EQ(evalChebyshevPolynomial(0.5, coefficients), 2.25);
  EXPECT_EQ(evalChebyshevPolynomial(1.0, coefficients), 3.5);
  EXPECT_EQ(evalChebyshevPolynomial(-1.0, coefficients), -1.5);
  EXPECT_EQ(evalMultiplicativeDepth(1.0, coefficients), 0);
}

TEST(PatersonStockmeyerChebyshevPolynomialEvaluation, QuarticPolynomial) {
  std::vector<double> coefficients = {3.0, -2, 0.5};  // 3T_0 - 2T_1 + 0.5T_2
  EXPECT_EQ(evalChebyshevPolynomial(0.0, coefficients), 2.5);
  EXPECT_EQ(evalChebyshevPolynomial(0.5, coefficients), 1.75);
  EXPECT_EQ(evalChebyshevPolynomial(1.0, coefficients), 1.5);
  EXPECT_EQ(evalChebyshevPolynomial(-1.0, coefficients), 5.5);
  EXPECT_EQ(evalMultiplicativeDepth(1.0, coefficients), 1);
}

TEST(PatersonStockmeyerChebyshevPolynomialEvaluation,
     Degree10ReluApproximation) {
  std::vector<double> coefficients = {
      0.3181208873635371,    0.5,
      0.21259683947738586,   -5.867410479938111e-17,
      -0.042871530276047946, -1.8610028415987085e-17,
      0.018698658506314733,  -4.6179699483612605e-17,
      -0.010761656553922883, -3.366722265465115e-17,
      0.016587575708296047};
  EXPECT_NEAR(evalChebyshevPolynomial(0.4, coefficients), 0.41137431561965,
              1e-14);
  EXPECT_EQ(evalMultiplicativeDepth(1.0, coefficients), 4);
}

TEST(PatersonStockmeyerChebyshevPolynomialEvaluation,
     Degree30ReluApproximation) {
  std::vector<double> coefficients = {
      0.31832346675681017,     0.49999999999999983,     0.21217958519441493,
      -1.5283153202438443e-16, -0.04241478265549799,    -1.6757847016582454e-16,
      0.018163396015548176,    -1.767747536761081e-16,  -0.010080473941797763,
      -1.5149547866410145e-16, 0.006407410077353227,    -1.7241739786587125e-16,
      -0.004430711645037601,   -1.945084334955229e-16,  0.003245919964971482,
      -1.6518278704532327e-16, -0.002480658725037781,   -1.9595604012841478e-16,
      0.0019585987170031,      -2.1192950463427032e-16, -0.0015874524581071755,
      -1.6449848029312714e-16, 0.0013151549048825073,   -1.8269203263988766e-16,
      -0.001110595707275929,   -1.8334855744042313e-16, 0.0009543608481579543,
      -1.5367947328521623e-16, -0.0008339720562323246,  -1.933105011429556e-16,
      0.0038349915376337043};
  EXPECT_NEAR(evalChebyshevPolynomial(0.7, coefficients), 0.7013677694556697,
              1e-14);
  EXPECT_EQ(evalMultiplicativeDepth(0.5, coefficients), 6);
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
