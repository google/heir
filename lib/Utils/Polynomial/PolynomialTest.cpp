
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Polynomial/Polynomial.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

TEST(PolynomialTest, TestDouble) {
  FloatPolynomial polynomial = FloatPolynomial::fromCoefficients({1., 2., 3.});
  FloatPolynomial result = polynomial.add(polynomial);
  int degree = 0;
  for (const FloatMonomial& term : result.getTerms()) {
    EXPECT_EQ(term.getCoefficient(), APFloat(2.0 * (1 + degree)));
    ++degree;
  }
}

TEST(PolynomialTest, TestAddEliminatingTerms) {
  IntPolynomial polynomial = IntPolynomial::fromCoefficients({1, 2, 3});
  IntPolynomial negationOfOneTerm = IntPolynomial::fromCoefficients({1, -2, 3});
  IntPolynomial result = polynomial.add(negationOfOneTerm);
  IntPolynomial expected = IntPolynomial::fromCoefficients({2, 0, 6});
  EXPECT_EQ(expected, result);
}

TEST(PolynomialTest, TestMonomialMul) {
  FloatPolynomial polynomial = FloatPolynomial::fromCoefficients({1., 2., 3.});
  FloatPolynomial result = polynomial.monomialMul(2);
  int i = 0;
  for (const FloatMonomial& term : polynomial.getTerms()) {
    auto resultTerm = result.getTerms()[i];
    EXPECT_EQ(term.getCoefficient(), resultTerm.getCoefficient());
    EXPECT_EQ(term.getExponent() + 2, resultTerm.getExponent());
    ++i;
  }
}

TEST(PolynomialTest, TestScale) {
  FloatPolynomial polynomial = FloatPolynomial::fromCoefficients({1., 2., 3.});
  FloatPolynomial result = polynomial.scale(APFloat(5.));
  int i = 0;
  for (const FloatMonomial& term : polynomial.getTerms()) {
    auto resultTerm = result.getTerms()[i];
    EXPECT_EQ(term.getCoefficient() * APFloat(5.), resultTerm.getCoefficient());
    EXPECT_EQ(term.getExponent(), resultTerm.getExponent());
    ++i;
  }
}

TEST(PolynomialTest, TestSub) {
  FloatPolynomial polynomial = FloatPolynomial::fromCoefficients({1., 2., 3.});
  FloatPolynomial subbed =
      FloatPolynomial::fromCoefficients({0., 0.5, 1., 2.0});
  FloatPolynomial result = polynomial.sub(subbed);
  FloatPolynomial expected =
      FloatPolynomial::fromCoefficients({1., 1.5, 2., -2.});
  EXPECT_EQ(expected, result);
}

TEST(PolynomialTest, TestAddUnequalTerms) {
  FloatPolynomial p1 = FloatPolynomial::fromCoefficients({1.});
  FloatPolynomial p2 = FloatPolynomial::fromCoefficients({0., 1.});
  FloatPolynomial result = p2.add(p1);
  FloatPolynomial expected = FloatPolynomial::fromCoefficients({1., 1.});
  EXPECT_EQ(expected, result);
}

TEST(PolynomialTest, TestAddUnequalTermsOtherWay) {
  FloatPolynomial p1 = FloatPolynomial::fromCoefficients({1.});
  FloatPolynomial p2 = FloatPolynomial::fromCoefficients({0., 1.});
  FloatPolynomial result = p1.add(p2);
  FloatPolynomial expected = FloatPolynomial::fromCoefficients({1., 1.});
  EXPECT_EQ(expected, result);
}

TEST(PolynomialTest, TestSubZero) {
  IntPolynomial polynomial = IntPolynomial::fromCoefficients({1, 2, 3});
  IntPolynomial result = polynomial.sub(polynomial);
  EXPECT_TRUE(result.isZero());
}

TEST(PolynomialTest, TestNaiveMul) {
  IntPolynomial polynomial = IntPolynomial::fromCoefficients({1, 2, 3});
  IntPolynomial result = polynomial.naiveMul(polynomial);
  IntPolynomial expected = IntPolynomial::fromCoefficients({1, 4, 10, 12, 9});
  EXPECT_EQ(expected, result);
}

TEST(PolynomialTest, TestNaiveMulFloat) {
  FloatPolynomial polynomial = FloatPolynomial::fromCoefficients({1., 2., 3.});
  FloatPolynomial result = polynomial.naiveMul(polynomial);
  FloatPolynomial expected =
      FloatPolynomial::fromCoefficients({1., 4., 10., 12., 9.});
  EXPECT_EQ(expected, result);
}

TEST(PolynomialTest, TestComposeFloat) {
  FloatPolynomial p1 = FloatPolynomial::fromCoefficients({-2.0, 1.0, 3.5});
  FloatPolynomial p2 = FloatPolynomial::fromCoefficients({1.0, 2.0, 0.0, 4.9});
  FloatPolynomial result = p1.compose(p2);
  FloatPolynomial expected = FloatPolynomial::fromCoefficients(
      {2.5, 16.0, 14.0, 39.200000000000003, 68.600000000000009, 0.0,
       84.03500000000001});
  EXPECT_EQ(expected, result);
}

TEST(PolynomialTest, TestComposeSkippingDegree) {
  SmallVector<IntMonomial> monomials;
  // 1 + 4x^3
  monomials.emplace_back(1, 0);
  monomials.emplace_back(4, 3);
  IntPolynomial p1 = IntPolynomial::fromMonomials(monomials).value();

  SmallVector<IntMonomial> monomials2;
  // -1 + 3x^2
  monomials2.emplace_back(-1, 0);
  monomials2.emplace_back(3, 2);
  IntPolynomial p2 = IntPolynomial::fromMonomials(monomials2).value();
  IntPolynomial result = p1.compose(p2);

  SmallVector<IntMonomial> expectedMonomials;
  // -3 + 36 x^2 - 108 x^4 + 108 x^6
  expectedMonomials.emplace_back(-3, 0);
  expectedMonomials.emplace_back(36, 2);
  expectedMonomials.emplace_back(-108, 4);
  expectedMonomials.emplace_back(108, 6);
  IntPolynomial expected =
      IntPolynomial::fromMonomials(expectedMonomials).value();
  result.dump();
  EXPECT_EQ(expected, result);
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
