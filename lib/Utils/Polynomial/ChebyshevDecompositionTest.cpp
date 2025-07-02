#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Polynomial/ChebyshevDecomposition.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

using ::llvm::SmallVector;
using ::testing::ElementsAre;

TEST(ChebyshevDecompositionTest, EmptyPolynomial) {
  ChebyshevBasisPolynomial p;
  ChebyshevDecomposition decomposition = decompose(p, 1);
  EXPECT_EQ(decomposition.generatorDegree, 1);
  EXPECT_TRUE(decomposition.coeffs.empty());
}

TEST(ChebyshevDecompositionTest, ConstantPolynomial) {
  ChebyshevBasisPolynomial p = {5.0};
  ChebyshevDecomposition decomposition = decompose(p, 1);
  ASSERT_EQ(decomposition.coeffs.size(), 1);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(5.0));
}

TEST(ChebyshevDecompositionTest, LinearPolynomialK1) {
  ChebyshevBasisPolynomial p = {-1.0, -3.0};
  ChebyshevDecomposition decomposition = decompose(p, 1);
  ASSERT_EQ(decomposition.coeffs.size(), 2);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(-1.0));
  EXPECT_THAT(decomposition.coeffs[1], ElementsAre(-3.0));
}

TEST(ChebyshevDecompositionTest, LinearPolynomialK3) {
  ChebyshevBasisPolynomial p = {-1.0, -3.0};
  ChebyshevDecomposition decomposition = decompose(p, 3);
  EXPECT_EQ(decomposition.generatorDegree, 3);
  ASSERT_EQ(decomposition.coeffs.size(), 1);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(-1.0, -3.0));
}

TEST(ChebyshevDecompositionTest, QuadraticPolynomial) {
  ChebyshevBasisPolynomial p = {1.0, -2.0, 3.0};
  ChebyshevDecomposition decomposition = decompose(p, 2);
  ASSERT_EQ(decomposition.coeffs.size(), 2);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(1.0, -2.0));
  EXPECT_THAT(decomposition.coeffs[1], ElementsAre(3.0));
}

// The expected output was found with the reference Python implementation and it
// was verified independently by evaluating the polynomials on one of the
// points.
TEST(ChebyshevDecompositionTest, Degree7Polynomial) {
  ChebyshevBasisPolynomial p = {1.0, -2.0, 3.0, 4.0, 5.0, 6.0, -7.0, 8.0};
  ChebyshevDecomposition decomposition = decompose(p, 3);
  EXPECT_EQ(decomposition.generatorDegree, 3);
  ASSERT_EQ(decomposition.coeffs.size(), 3);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(8.0, -16.0, -2.0));
  EXPECT_THAT(decomposition.coeffs[1], ElementsAre(4.0, 10.0, -4.0));
  EXPECT_THAT(decomposition.coeffs[2], ElementsAre(-14.0, 32.0));
}

TEST(ChebyshevDecompositionTest, Degree20Polynomial) {
  ChebyshevBasisPolynomial p = {-1.0, 2.0,   3.0,  4.0,  5.0,  -6.0,  7.0,
                                -8.0, 9.0,   10.0, 11.0, 12.0, -13.0, 14.0,
                                15.0, -16.0, 17.0, 18.0, 19.0, 20.0,  21.0};
  ChebyshevDecomposition decomposition = decompose(p, 4);
  EXPECT_EQ(decomposition.generatorDegree, 4);
  ASSERT_EQ(decomposition.coeffs.size(), 6);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(7.0, 2.0, 19.0, 32.0));
  EXPECT_THAT(decomposition.coeffs[1], ElementsAre(149.0, -12.0, 8.0, 100.0));
  EXPECT_THAT(decomposition.coeffs[2],
              ElementsAre(-118.0, -112.0, -244.0, -248.0));
  EXPECT_THAT(decomposition.coeffs[3],
              ElementsAre(-472.0, -48.0, -32.0, -272.0));
  EXPECT_THAT(decomposition.coeffs[4], ElementsAre(136.0, 288.0, 304.0, 320.0));
  EXPECT_THAT(decomposition.coeffs[5], ElementsAre(336.0));
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
