#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Polynomial/ChebyshevDecomposition.h"
#include "llvm/include/llvm/ADT/APFloat.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

using ::llvm::APFloat;
using ::llvm::SmallVector;
using ::testing::ElementsAre;

TEST(ChebyshevDecompositionTest, EmptyPolynomial) {
  ChebyshevBasisPolynomial p;
  ChebyshevDecomposition decomposition = decompose(p, 1);
  EXPECT_EQ(decomposition.generatorDegree, 1);
  EXPECT_TRUE(decomposition.coeffs.empty());
}

TEST(ChebyshevDecompositionTest, ConstantPolynomial) {
  ChebyshevBasisPolynomial p = {APFloat(5.0)};
  ChebyshevDecomposition decomposition = decompose(p, 1);
  ASSERT_EQ(decomposition.coeffs.size(), 1);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(APFloat(5.0)));
}

TEST(ChebyshevDecompositionTest, LinearPolynomialK1) {
  ChebyshevBasisPolynomial p = {APFloat(-1.0), APFloat(-3.0)};
  ChebyshevDecomposition decomposition = decompose(p, 1);
  ASSERT_EQ(decomposition.coeffs.size(), 2);
  EXPECT_THAT(decomposition.coeffs[0], ElementsAre(APFloat(-1.0)));
  EXPECT_THAT(decomposition.coeffs[1], ElementsAre(APFloat(-3.0)));
}

TEST(ChebyshevDecompositionTest, LinearPolynomialK3) {
  ChebyshevBasisPolynomial p = {APFloat(-1.0), APFloat(-3.0)};
  ChebyshevDecomposition decomposition = decompose(p, 3);
  EXPECT_EQ(decomposition.generatorDegree, 3);
  ASSERT_EQ(decomposition.coeffs.size(), 1);
  EXPECT_THAT(decomposition.coeffs[0],
              ElementsAre(APFloat(-1.0), APFloat(-3.0)));
}

TEST(ChebyshevDecompositionTest, QuadraticPolynomial) {
  ChebyshevBasisPolynomial p = {APFloat(1.0), APFloat(-2.0), APFloat(3.0)};
  ChebyshevDecomposition decomposition = decompose(p, 2);
  ASSERT_EQ(decomposition.coeffs.size(), 2);
  EXPECT_THAT(decomposition.coeffs[0],
              ElementsAre(APFloat(1.0), APFloat(-2.0)));
  EXPECT_THAT(decomposition.coeffs[1], ElementsAre(APFloat(3.0)));
}

// The expected output was found with the reference Python implementation and it
// was verified independently by evaluating the polynomials on one of the
// points.
TEST(ChebyshevDecompositionTest, Degree7Polynomial) {
  ChebyshevBasisPolynomial p = {APFloat(1.0),  APFloat(-2.0), APFloat(3.0),
                                APFloat(4.0),  APFloat(5.0),  APFloat(6.0),
                                APFloat(-7.0), APFloat(8.0)};
  ChebyshevDecomposition decomposition = decompose(p, 3);
  EXPECT_EQ(decomposition.generatorDegree, 3);
  ASSERT_EQ(decomposition.coeffs.size(), 3);
  EXPECT_THAT(decomposition.coeffs[0],
              ElementsAre(APFloat(8.0), APFloat(-16.0), APFloat(-2.0)));
  EXPECT_THAT(decomposition.coeffs[1],
              ElementsAre(APFloat(4.0), APFloat(10.0), APFloat(-4.0)));
  EXPECT_THAT(decomposition.coeffs[2],
              ElementsAre(APFloat(-14.0), APFloat(32.0)));
}

TEST(ChebyshevDecompositionTest, Degree20Polynomial) {
  ChebyshevBasisPolynomial p = {APFloat(-1.0),  APFloat(2.0),  APFloat(3.0),
                                APFloat(4.0),   APFloat(5.0),  APFloat(-6.0),
                                APFloat(7.0),   APFloat(-8.0), APFloat(9.0),
                                APFloat(10.0),  APFloat(11.0), APFloat(12.0),
                                APFloat(-13.0), APFloat(14.0), APFloat(15.0),
                                APFloat(-16.0), APFloat(17.0), APFloat(18.0),
                                APFloat(19.0),  APFloat(20.0), APFloat(21.0)};
  ChebyshevDecomposition decomposition = decompose(p, 4);
  EXPECT_EQ(decomposition.generatorDegree, 4);
  ASSERT_EQ(decomposition.coeffs.size(), 6);
  EXPECT_THAT(
      decomposition.coeffs[0],
      ElementsAre(APFloat(7.0), APFloat(2.0), APFloat(19.0), APFloat(32.0)));
  EXPECT_THAT(decomposition.coeffs[1],
              ElementsAre(APFloat(149.0), APFloat(-12.0), APFloat(8.0),
                          APFloat(100.0)));
  EXPECT_THAT(decomposition.coeffs[2],
              ElementsAre(APFloat(-118.0), APFloat(-112.0), APFloat(-244.0),
                          APFloat(-248.0)));
  EXPECT_THAT(decomposition.coeffs[3],
              ElementsAre(APFloat(-472.0), APFloat(-48.0), APFloat(-32.0),
                          APFloat(-272.0)));
  EXPECT_THAT(decomposition.coeffs[4],
              ElementsAre(APFloat(136.0), APFloat(288.0), APFloat(304.0),
                          APFloat(320.0)));
  EXPECT_THAT(decomposition.coeffs[5], ElementsAre(APFloat(336.0)));
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
