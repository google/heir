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
using ::mlir::heir::polynomial::FloatPolynomial;
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
  getChebyshevPolynomials(n, chebPolys);

  for (const auto& p : chebPolys) p.dump();

  EXPECT_THAT(
      chebPolys,
      ElementsAre(
          FloatPolynomial::fromCoefficients({1.}),
          FloatPolynomial::fromCoefficients({0., 2.}),
          FloatPolynomial::fromCoefficients({-1., 0., 4.}),
          FloatPolynomial::fromCoefficients({0., -4., 0., 8.}),
          FloatPolynomial::fromCoefficients({1., 0., -12., 0., 16.}),
          FloatPolynomial::fromCoefficients({0., 6., 0., -32., 0., 32.}),
          FloatPolynomial::fromCoefficients({-1., 0., 24., 0., -80., 0., 64.}),
          FloatPolynomial::fromCoefficients(
              {0., -8., 0., 80., 0., -192., 0., 128.}),
          FloatPolynomial::fromCoefficients(
              {1., 0., -40., 0., 240., 0., -448., 0., 256.})));
}

}  // namespace
}  // namespace approximation
}  // namespace heir
}  // namespace mlir
