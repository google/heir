#include <cmath>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Approximation/CaratheodoryFejer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define EPSILON 1e-11

namespace mlir {
namespace heir {
namespace approximation {
namespace {

using ::llvm::APFloat;
using ::mlir::heir::polynomial::FloatPolynomial;
using ::testing::DoubleNear;

TEST(CaratheodoryFejerTest, ApproximateExpDegree3) {
  auto func = [](const APFloat& x) {
    return APFloat(std::exp(x.convertToDouble()));
  };
  FloatPolynomial actual = caratheodoryFejerApproximation(func, 3);
  // Values taken from reference impl are exact.
  FloatPolynomial expected = FloatPolynomial::fromCoefficients(
      {0.9945811640427066, 0.9956553725361579, 0.5429702814725632,
       0.1795458211087378});
  EXPECT_EQ(actual, expected);
}

TEST(CaratheodoryFejerTest, ApproximateReluDegree14) {
  auto relu = [](const APFloat& x) {
    APFloat zero = APFloat::getZero(x.getSemantics());
    return x > zero ? x : zero;
  };
  FloatPolynomial actual = caratheodoryFejerApproximation(relu, 14);

  // The reference implementation prints coefficients that are ~1e-12 away from
  // our implementation, mainly because the eigenvalue solver details are
  // slightly different. For instance, the max eigenvalue in our implementation
  // is
  //
  //
  // -0.415033778742867843
  // 0.41503377874286651
  // 0.358041674766206519
  // -0.358041674766206408
  // -0.302283813201297547
  // 0.302283813201297602
  // 0.244610415109838275
  // -0.244610415109838636
  // -0.182890410854375879
  // 0.182890410854376101
  // 0.115325658064263425
  // -0.115325658064263148
  // -0.0399306015339729384
  // 0.0399306015339727718
  //
  // But in the reference implementation, the basis elements are permuted so
  // that the adjacent positive values and negative values are swapped. This is
  // still a valid eigenvector, but it results in slight difference in floating
  // point error accumulation as the rest of the algorithm proceeds.
  auto terms = actual.getTerms();
  EXPECT_THAT(terms[0].getCoefficient().convertToDouble(),
              DoubleNear(0.010384627976349288, EPSILON));
  EXPECT_THAT(terms[1].getCoefficient().convertToDouble(),
              DoubleNear(0.4999999999999994, EPSILON));
  EXPECT_THAT(terms[2].getCoefficient().convertToDouble(),
              DoubleNear(3.227328667600437, EPSILON));
  EXPECT_THAT(terms[3].getCoefficient().convertToDouble(),
              DoubleNear(2.1564570993799688e-14, EPSILON));
  EXPECT_THAT(terms[4].getCoefficient().convertToDouble(),
              DoubleNear(-27.86732536231614, EPSILON));
  EXPECT_THAT(terms[5].getCoefficient().convertToDouble(),
              DoubleNear(-1.965772591254676e-13, EPSILON));
  EXPECT_THAT(terms[6].getCoefficient().convertToDouble(),
              DoubleNear(139.12944753041404, EPSILON));
  EXPECT_THAT(terms[7].getCoefficient().convertToDouble(),
              DoubleNear(7.496488571843804e-13, EPSILON));
  EXPECT_THAT(terms[8].getCoefficient().convertToDouble(),
              DoubleNear(-363.6062351528312, EPSILON));
  EXPECT_THAT(terms[9].getCoefficient().convertToDouble(),
              DoubleNear(-1.3773783921527744e-12, EPSILON));
  EXPECT_THAT(terms[10].getCoefficient().convertToDouble(),
              DoubleNear(505.9489721657369, EPSILON));
  EXPECT_THAT(terms[11].getCoefficient().convertToDouble(),
              DoubleNear(1.2076732984649801e-12, EPSILON));
  EXPECT_THAT(terms[12].getCoefficient().convertToDouble(),
              DoubleNear(-355.4120699445272, EPSILON));
  EXPECT_THAT(terms[13].getCoefficient().convertToDouble(),
              DoubleNear(-4.050490139246503e-13, EPSILON));
  EXPECT_THAT(terms[14].getCoefficient().convertToDouble(),
              DoubleNear(99.07988219049058, EPSILON));
}

}  // namespace
}  // namespace approximation
}  // namespace heir
}  // namespace mlir
