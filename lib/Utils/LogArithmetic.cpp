#include "lib/Utils/LogArithmetic.h"

#include <cmath>
#include <utility>

namespace mlir {
namespace heir {

Log2Arithmetic Log2Arithmetic::operator+(const Log2Arithmetic &rhs) const {
  // give sum of two log value
  double log2a = getLog2Value();
  double log2b = rhs.getLog2Value();
  // make a >= b
  if (log2b > log2a) {
    std::swap(log2a, log2b);
  }
  // if a >>> b, do not call std::exp2 to avoid overflow
  if (log2b == NEGATIVE_INFINITY || log2a - log2b > 512.0) {
    // Use internal constructor to avoid confusion with double
    return Log2Arithmetic(log2a);
  }
  // log2(a + b) = log2(a) + log2(1 + pow(2, (log2(b) - log2(a))))
  // More numerically stable than direct computation
  double log2Sum;
  if (log2a == log2b) {
    // Special case when equal: log2(2a) = 1 + log2a
    log2Sum = 1.0 + log2a;
  } else {
    double exponent = log2b - log2a;
    // For small exponents, use more accurate approximation
    log2Sum = log2a + std::log1p(std::exp2(exponent)) / std::log(2);
  }
  // Use internal constructor to avoid confusion with double
  return Log2Arithmetic(log2Sum);
}

Log2Arithmetic Log2Arithmetic::operator*(const Log2Arithmetic &rhs) const {
  // give product of two log value
  auto log2a = getLog2Value();
  auto log2b = rhs.getLog2Value();
  // log2(a * b) = log2(a) + log2(b)
  // this works for negative infinity
  return Log2Arithmetic(log2a + log2b);
}

}  // namespace heir
}  // namespace mlir
