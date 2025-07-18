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
using polynomial::FloatPolynomial;
using ::testing::DoubleNear;

TEST(CaratheodoryFejerTest, ApproximateExpDegree3) {
  auto func = [](const APFloat& x) {
    return APFloat(std::exp(x.convertToDouble()));
  };
  FloatPolynomial actual = caratheodoryFejerApproximation(func, 3);

  auto terms = actual.getTerms();
  EXPECT_THAT(terms[0].getCoefficient().convertToDouble(),
              DoubleNear(0.9945794763246951, EPSILON));
  EXPECT_THAT(terms[1].getCoefficient().convertToDouble(),
              DoubleNear(0.9956677100276301, EPSILON));
  EXPECT_THAT(terms[2].getCoefficient().convertToDouble(),
              DoubleNear(0.5429727883818608, EPSILON));
  EXPECT_THAT(terms[3].getCoefficient().convertToDouble(),
              DoubleNear(0.17953348361617388, EPSILON));
}

TEST(CaratheodoryFejerTest, ApproximateExpDegree3MinusTwoTwoInterval) {
  auto func = [](const APFloat& x) {
    return APFloat(std::exp(x.convertToDouble()));
  };
  FloatPolynomial actual = caratheodoryFejerApproximation(func, 3, -2.0, 2.0);
  auto terms = actual.getTerms();
  EXPECT_THAT(terms[0].getCoefficient().convertToDouble(),
              DoubleNear(0.9023129365897373, EPSILON));
  EXPECT_THAT(terms[1].getCoefficient().convertToDouble(),
              DoubleNear(0.9221474912559928, EPSILON));
  EXPECT_THAT(terms[2].getCoefficient().convertToDouble(),
              DoubleNear(0.688635050076054, EPSILON));
  EXPECT_THAT(terms[3].getCoefficient().convertToDouble(),
              DoubleNear(0.2228206781698768, EPSILON));
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
              DoubleNear(0.012463964097674111, EPSILON));
  EXPECT_THAT(terms[1].getCoefficient().convertToDouble(),
              DoubleNear(0.49999999999999944, EPSILON));
  EXPECT_THAT(terms[2].getCoefficient().convertToDouble(),
              DoubleNear(3.1497214624871868, EPSILON));
  EXPECT_THAT(terms[3].getCoefficient().convertToDouble(),
              DoubleNear(1.125613084629512e-14, EPSILON));
  EXPECT_THAT(terms[4].getCoefficient().convertToDouble(),
              DoubleNear(-26.28684650927164, EPSILON));
  EXPECT_THAT(terms[5].getCoefficient().convertToDouble(),
              DoubleNear(-8.710818612582302e-14, EPSILON));
  EXPECT_THAT(terms[6].getCoefficient().convertToDouble(),
              DoubleNear(128.21098801816572, EPSILON));
  EXPECT_THAT(terms[7].getCoefficient().convertToDouble(),
              DoubleNear(2.9771289412372216e-13, EPSILON));
  EXPECT_THAT(terms[8].getCoefficient().convertToDouble(),
              DoubleNear(-329.7013599574576, EPSILON));
  EXPECT_THAT(terms[9].getCoefficient().convertToDouble(),
              DoubleNear(-4.950876525219969e-13, EPSILON));
  EXPECT_THAT(terms[10].getCoefficient().convertToDouble(),
              DoubleNear(453.4426992698845, EPSILON));
  EXPECT_THAT(terms[11].getCoefficient().convertToDouble(),
              DoubleNear(3.9333787272256463e-13, EPSILON));
  EXPECT_THAT(terms[12].getCoefficient().convertToDouble(),
              DoubleNear(-315.7369991939929, EPSILON));
  EXPECT_THAT(terms[13].getCoefficient().convertToDouble(),
              DoubleNear(-1.195396993539409e-13, EPSILON));
  EXPECT_THAT(terms[14].getCoefficient().convertToDouble(),
              DoubleNear(87.41796048301983, EPSILON));
}

TEST(CaratheodoryFejerTest, ReluDegree3) {
  // Regression test for https://github.com/google/heir/issues/1609
  auto relu = [](const APFloat& x) {
    APFloat zero = APFloat::getZero(x.getSemantics());
    return x > zero ? x : zero;
  };
  FloatPolynomial actual = caratheodoryFejerApproximation(relu, 3);

  auto terms = actual.getTerms();
  EXPECT_THAT(terms[0].getCoefficient().convertToDouble(),
              DoubleNear(0.06972184658933259, EPSILON));
  EXPECT_THAT(terms[1].getCoefficient().convertToDouble(),
              DoubleNear(0.5, EPSILON));
  EXPECT_THAT(terms[2].getCoefficient().convertToDouble(),
              DoubleNear(0.48793143684847234, EPSILON));
  EXPECT_THAT(terms[3].getCoefficient().convertToDouble(),
              DoubleNear(-2.7192355874675714e-17, EPSILON));
}

}  // namespace
}  // namespace approximation
}  // namespace heir
}  // namespace mlir
