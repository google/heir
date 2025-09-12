#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Approximation/Chebyshev.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace approximation {
namespace {

using ::llvm::APFloat;
using polynomial::FloatPolynomial;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::ElementsAre;

TEST(ChebyshevTest, TestGetChebyshevPointsSingle) {
  SmallVector<APFloat> chebPts;
  int64_t n = 1;
  getChebyshevPoints(n, chebPts);
  EXPECT_THAT(chebPts, ElementsAre(APFloat(0.)));
}

TEST(ChebyshevTest, TestGetChebyshevPoints5) {
  SmallVector<APFloat> chebPts;
  int64_t n = 5;
  getChebyshevPoints(n, chebPts);
  EXPECT_THAT(chebPts, ElementsAre(APFloat(-1.0), APFloat(-0.7071067811865475),
                                   APFloat(0.0), APFloat(0.7071067811865475),
                                   APFloat(1.0)));
}

TEST(ChebyshevTest, TestGetChebyshevPoints9) {
  SmallVector<APFloat> chebPts;
  int64_t n = 9;
  getChebyshevPoints(n, chebPts);
  EXPECT_THAT(chebPts, ElementsAre(APFloat(-1.0), APFloat(-0.9238795325112867),
                                   APFloat(-0.7071067811865475),
                                   APFloat(-0.3826834323650898), APFloat(0.0),
                                   APFloat(0.3826834323650898),
                                   APFloat(0.7071067811865475),
                                   APFloat(0.9238795325112867), APFloat(1.0)));
}

TEST(ChebyshevTest, TestGetChebyshevPolynomials) {
  SmallVector<FloatPolynomial> chebPolys;
  int64_t n = 9;
  chebPolys.reserve(n);
  polynomial::ChebyshevPolynomial::getChebyshevPolynomials(n, chebPolys);
  EXPECT_THAT(
      chebPolys,
      ElementsAre(
          FloatPolynomial::fromCoefficients({1.}),
          FloatPolynomial::fromCoefficients({0., 1.}),
          FloatPolynomial::fromCoefficients({-1., 0., 2.}),
          FloatPolynomial::fromCoefficients({0., -3., 0., 4.}),
          FloatPolynomial::fromCoefficients({1., 0., -8., 0., 8.}),
          FloatPolynomial::fromCoefficients({0., 5., 0., -20., 0., 16.}),
          FloatPolynomial::fromCoefficients({-1., 0., 18., 0., -48., 0., 32.}),
          FloatPolynomial::fromCoefficients(
              {0., -7., 0., 56., 0., -112., 0., 64.}),
          FloatPolynomial::fromCoefficients(
              {1., 0., -32., 0., 160., 0., -256., 0., 128.})));
}

TEST(ChebyshevTest, TestChebyshevToMonomial) {
  // 1 (1) - 1 (-1 + 2x^2) + 2 (-3x + 4x^3)
  SmallVector<APFloat> chebCoeffs = {APFloat(1.0), APFloat(0.0), APFloat(-1.0),
                                     APFloat(2.0)};
  polynomial::ChebyshevPolynomial chebPoly =
      polynomial::ChebyshevPolynomial(chebCoeffs);
  // 2 - 6 x - 2 x^2 + 8 x^3
  FloatPolynomial expected =
      FloatPolynomial::fromCoefficients({2.0, -6.0, -2.0, 8.0});
  FloatPolynomial actual = chebPoly.toStandardBasis();
  EXPECT_EQ(actual, expected);
}

TEST(ChebyshevTest, TestInterpolateChebyshevExpDegree3) {
  // degree 3 implies we need 4 points.
  SmallVector<APFloat> chebPts = {APFloat(-1.0), APFloat(-0.5), APFloat(0.5),
                                  APFloat(1.0)};
  SmallVector<APFloat> expVals;
  expVals.reserve(chebPts.size());
  for (const APFloat& pt : chebPts) {
    expVals.push_back(APFloat(std::exp(pt.convertToDouble())));
  }

  SmallVector<APFloat> actual;
  interpolateChebyshev(expVals, actual);

  EXPECT_THAT(actual[0].convertToDouble(), DoubleEq(1.2661108550760016));
  EXPECT_THAT(actual[1].convertToDouble(), DoubleEq(1.1308643327583656));
  EXPECT_THAT(actual[2].convertToDouble(), DoubleEq(0.276969779739242));
  // This test is slightly off from what numpy produces (up to ~10^{-15}), not
  // sure why.
  // EXPECT_THAT(actual[3].convertToDouble(), DoubleEq(0.04433686088543568));
  //
  // This is also slightly off from what is produced on MacOS platforms by the
  // same code. EXPECT_THAT(actual[3].convertToDouble(),
  // DoubleEq(0.044336860885435571));
  //
  // EXPECT_THAT(actual[3].convertToDouble(), DoubleEq(0.044336860885435536));
  EXPECT_THAT(actual[3].convertToDouble(),
              DoubleNear(0.044336860885435536, 1e-14));
}

TEST(ChebyshevTest, ExpAutoSelectDegree) {
  auto func = [](const APFloat& x) {
    return APFloat(std::exp(x.convertToDouble()));
  };
  SmallVector<APFloat> coeffs;
  interpolateChebyshevWithSmartDegreeSelection(func, coeffs);
  polynomial::ChebyshevPolynomial chebPoly =
      polynomial::ChebyshevPolynomial(coeffs);
  FloatPolynomial result = chebPoly.toStandardBasis();

  // approximate infinity norm error on [-1, 1]
  double error = 0.0;
  int N = 100;
  for (int i = 0; i < N; ++i) {
    double input = 2.0 * i / (double)N - 1.0;
    double expected = std::exp(input);
    double actual = 0;
    for (auto& term : result.getTerms()) {
      actual += term.getCoefficient().convertToDouble() *
                std::pow(input, term.getExponent().getZExtValue());
    }
    error = std::max(error, std::abs(expected - actual));
  }
  EXPECT_LT(error, 1e-14);
}

}  // namespace
}  // namespace approximation
}  // namespace heir
}  // namespace mlir
