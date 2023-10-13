#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "include/Dialect/Poly/IR/Polynomial.h"
#include "include/Dialect/Poly/IR/PolynomialDetail.h"
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/Support/StorageUniquer.h"  // from @llvm-project

namespace mlir::heir::poly {
namespace {

using ::testing::ElementsAre;

TEST(PolynomialTest, TestBuilder) {
  mlir::MLIRContext context;
  context.getAttributeUniquer()
      .registerParametricStorageType<detail::PolynomialStorage>();
  auto poly = Polynomial::fromCoefficients({1, 2, 3}, &context);
  std::string result;
  llvm::raw_string_ostream stream(result);
  poly.print(stream);

  EXPECT_EQ(result, "1 + 2x + 3x**2");
}

TEST(PolynomialTest, TestBuilderFromMonomials) {
  mlir::MLIRContext context;
  context.getAttributeUniquer()
      .registerParametricStorageType<detail::PolynomialStorage>();
  std::vector<Monomial> monomials;
  monomials.push_back(Monomial(1, 1024));
  monomials.push_back(Monomial(1, 0));
  auto poly = Polynomial::fromMonomials(monomials, &context);

  std::string result;
  llvm::raw_string_ostream stream(result);
  poly.print(stream);

  EXPECT_EQ(result, "1 + x**1024");
}

TEST(PolynomialTest, TestSortedDegree) {
  mlir::MLIRContext context;
  context.getAttributeUniquer()
      .registerParametricStorageType<detail::PolynomialStorage>();
  std::vector<Monomial> monomials;
  monomials.push_back(Monomial(1, 9));
  monomials.push_back(Monomial(1, 1));
  monomials.push_back(Monomial(1, 3));
  monomials.push_back(Monomial(1, 0));

  auto poly = Polynomial::fromMonomials(monomials, &context);

  std::vector<uint64_t> actualDegrees;
  for (auto term : poly.getTerms()) {
    actualDegrees.push_back(term.exponent.getZExtValue());
  }

  EXPECT_THAT(actualDegrees, ElementsAre(0, 1, 3, 9));
}

TEST(PolynomialTest, TestToIdentifier) {
  mlir::MLIRContext context;
  context.getAttributeUniquer()
      .registerParametricStorageType<detail::PolynomialStorage>();
  auto poly = Polynomial::fromCoefficients({1, 2, 3}, &context);
  std::string result = poly.toIdentifier();
  EXPECT_EQ(result, "1_2x_3x2");
}

TEST(PolynomialTest, TestToIdentifier_WithMinus) {
  mlir::MLIRContext context;
  context.getAttributeUniquer()
      .registerParametricStorageType<detail::PolynomialStorage>();
  auto poly = Polynomial::fromCoefficients({1, -2, 3}, &context);
  std::string result = poly.toIdentifier();
  EXPECT_EQ(result, "1_-2x_3x2");
}

}  // namespace
}  // namespace mlir::heir::poly
