
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Polynomial/Polynomial.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

TEST(PolynomialTest, TestDouble) {
  SmallVector<double> coeffs;
  coeffs.push_back(1.0);
  coeffs.push_back(2.0);
  coeffs.push_back(3.0);
  FloatPolynomial polynomial = FloatPolynomial::fromCoefficients(coeffs);
  FloatPolynomial result = polynomial.add(polynomial);

  int degree = 0;
  for (const FloatMonomial &term : result.getTerms()) {
    EXPECT_EQ(term.getCoefficient(), APFloat(2.0 * (1 + degree)));
    ++degree;
  }
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
