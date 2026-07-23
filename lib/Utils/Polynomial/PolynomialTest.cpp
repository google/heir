
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSAttributes.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Polynomial/RNSPolynomial.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

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

TEST(PolynomialTest, GetDenseCoefficientVector) {
  SmallVector<IntMonomial> monomials;
  monomials.emplace_back(1, 0);
  IntMonomial narrowCoefficient;
  narrowCoefficient.setCoefficient(APInt(8, -3, true));
  narrowCoefficient.setExponent(APInt(apintBitWidth, 2));
  monomials.push_back(narrowCoefficient);
  IntPolynomial polynomial = IntPolynomial::fromMonomials(monomials).value();

  SmallVector<APInt> coefficients = polynomial.getDenseCoefficientVector();

  ASSERT_EQ(coefficients.size(), 3);
  EXPECT_EQ(coefficients[0], APInt(apintBitWidth, 1));
  EXPECT_EQ(coefficients[1], APInt(apintBitWidth, 0));
  EXPECT_EQ(coefficients[2], APInt(apintBitWidth, -3, true));
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

TEST(RNSPolynomialTest, Arithmetic) {
  SmallVector<uint64_t> moduli = {17, 13};
  SmallVector<uint64_t> coeffs1 = {1, 2, 3, 4, 5, 6};
  SmallVector<uint64_t> coeffs2 = {16, 15, 14, 12, 11, 10};

  RNSPolynomial poly1(coeffs1, moduli);
  RNSPolynomial poly2(coeffs2, moduli);

  RNSPolynomial sum = poly1.add(poly2);
  SmallVector<uint64_t> expectedSum = {0, 0, 0, 3, 3, 3};
  EXPECT_EQ(sum.getData(), llvm::ArrayRef<uint64_t>(expectedSum));

  RNSPolynomial diff = poly1.sub(poly2);
  SmallVector<uint64_t> expectedDiff = {2, 4, 6, 5, 7, 9};
  EXPECT_EQ(diff.getData(), llvm::ArrayRef<uint64_t>(expectedDiff));
}

TEST(RNSPolynomialTest, ScalarMul) {
  RNSPolynomial poly({1, 2, 3, 4, 5, 6}, {17, 13});

  std::optional<RNSPolynomial> result = poly.scalarMul({3, 15});

  ASSERT_TRUE(result.has_value());
  SmallVector<uint64_t> expected = {3, 6, 9, 8, 10, 12};
  EXPECT_EQ(result->getData(), llvm::ArrayRef<uint64_t>(expected));
}

TEST(RNSPolynomialTest, TestRepresentation) {
  SmallVector<uint64_t> moduli = {17, 13};
  SmallVector<uint64_t> coeffs = {1, 2, 3, 4, 5, 6};

  // Default representation should be Coefficient
  RNSPolynomial polyCoeff(coeffs, moduli);
  EXPECT_EQ(polyCoeff.getRepresentation(), Form::COEFF);
  EXPECT_FALSE(polyCoeff.isNtt());

  // Explicit NTT representation
  RNSPolynomial polyNtt(coeffs, moduli, Form::EVAL);
  EXPECT_EQ(polyNtt.getRepresentation(), Form::EVAL);
  EXPECT_TRUE(polyNtt.isNtt());

  // Test that adding/subtracting mismatched representations asserts
  EXPECT_DEBUG_DEATH(polyCoeff.add(polyNtt),
                     "Representations must match for arithmetic");
  EXPECT_DEBUG_DEATH(polyCoeff.sub(polyNtt),
                     "Representations must match for arithmetic");
}

TEST(RNSPolynomialTest, TestConversions) {
  SmallVector<uint64_t> moduli = {17, 41};
  SmallVector<uint64_t> coeffs = {1, 2, 3, 4, 5, 6, 7, 8};

  RNSPolynomial poly(coeffs, moduli);

  // Test round-trip toNtt -> toCoefficient
  RNSPolynomial ntt = poly.toNtt();
  EXPECT_TRUE(ntt.isNtt());

  RNSPolynomial roundtrip = ntt.toCoefficient();
  EXPECT_FALSE(roundtrip.isNtt());
  EXPECT_EQ(roundtrip, poly);
}

TEST(RNSPolynomialTest, TestMul) {
  SmallVector<uint64_t> moduli = {17, 41};
  SmallVector<uint64_t> coeffs1 = {1, 2, 0, 0, 3, 4, 0, 0};
  SmallVector<uint64_t> coeffs2 = {5, 6, 0, 0, 7, 8, 0, 0};

  RNSPolynomial poly1(coeffs1, moduli);
  RNSPolynomial poly2(coeffs2, moduli);

  // Test multiplication in Coefficient form (uses NTT under the hood)
  RNSPolynomial prodCoeff = poly1.mul(poly2);
  EXPECT_FALSE(prodCoeff.isNtt());
  SmallVector<uint64_t> expectedProd = {5, 16, 12, 0, 21, 11, 32, 0};
  EXPECT_EQ(prodCoeff.getData(), llvm::ArrayRef<uint64_t>(expectedProd));

  // Test multiplication in NTT form
  RNSPolynomial ntt1 = poly1.toNtt();
  RNSPolynomial ntt2 = poly2.toNtt();
  RNSPolynomial prodNtt = ntt1.mul(ntt2);
  EXPECT_TRUE(prodNtt.isNtt());
  EXPECT_EQ(prodNtt.toCoefficient(), prodCoeff);
}

TEST(RNSPolynomialTest, TestPrecomputedRoots) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::heir::rns::RNSDialect>();
  context.loadDialect<mlir::heir::mod_arith::ModArithDialect>();

  SmallVector<uint64_t> moduli = {17, 41};
  SmallVector<uint64_t> coeffs = {1, 2, 3, 4, 5, 6, 7, 8};

  RNSPolynomial poly(coeffs, moduli);

  auto i64Type = mlir::IntegerType::get(&context, 64);
  auto mod17Type = mlir::heir::mod_arith::ModArithType::get(
      &context, mlir::IntegerAttr::get(i64Type, 17));
  auto mod41Type = mlir::heir::mod_arith::ModArithType::get(
      &context, mlir::IntegerAttr::get(i64Type, 41));

  auto rnsType =
      mlir::heir::rns::RNSType::get(&context, {mod17Type, mod41Type});

  // 1. Test with roots returned by findPrimitive2nthRoot (should match
  // on-the-fly)
  auto root16_17 = mlir::heir::findPrimitive2nthRoot(mlir::APInt(64, 17), 4);
  auto root16_41 = mlir::heir::findPrimitive2nthRoot(mlir::APInt(64, 41), 4);
  ASSERT_TRUE(root16_17.has_value());
  ASSERT_TRUE(root16_41.has_value());

  auto root17Attr = mlir::heir::mod_arith::ModArithAttr::get(
      &context, mod17Type,
      mlir::IntegerAttr::get(i64Type, root16_17->getZExtValue()));
  auto root41Attr = mlir::heir::mod_arith::ModArithAttr::get(
      &context, mod41Type,
      mlir::IntegerAttr::get(i64Type, root16_41->getZExtValue()));

  auto rnsAttr =
      mlir::heir::rns::RNSAttr::get(rnsType, {root17Attr, root41Attr});

  // Test toNtt with precomputed roots (matching on-the-fly)
  RNSPolynomial ntt = poly.toNtt(rnsAttr);
  EXPECT_TRUE(ntt.isNtt());

  // Compare with on-the-fly computation
  RNSPolynomial nttOnTheFly = poly.toNtt();
  EXPECT_EQ(ntt, nttOnTheFly);

  // Test toCoefficient with precomputed roots
  RNSPolynomial roundtrip = ntt.toCoefficient(rnsAttr);
  EXPECT_FALSE(roundtrip.isNtt());
  EXPECT_EQ(roundtrip, poly);

  // 2. Test with DIFFERENT valid roots (should round-trip, but might not match
  // on-the-fly) We know 9 is primitive 8-th root mod 17, and 3 is primitive
  // 8-th root mod 41.
  auto diffRoot17Attr = mlir::heir::mod_arith::ModArithAttr::get(
      &context, mod17Type, mlir::IntegerAttr::get(i64Type, 9));
  auto diffRoot41Attr = mlir::heir::mod_arith::ModArithAttr::get(
      &context, mod41Type, mlir::IntegerAttr::get(i64Type, 3));
  auto diffRnsAttr =
      mlir::heir::rns::RNSAttr::get(rnsType, {diffRoot17Attr, diffRoot41Attr});

  RNSPolynomial nttDiff = poly.toNtt(diffRnsAttr);
  EXPECT_TRUE(nttDiff.isNtt());

  RNSPolynomial roundtripDiff = nttDiff.toCoefficient(diffRnsAttr);
  EXPECT_FALSE(roundtripDiff.isNtt());
  EXPECT_EQ(roundtripDiff, poly);
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
