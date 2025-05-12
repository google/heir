#include "lib/Analysis/NoiseAnalysis/Noise.h"

#include <cmath>
#include <string>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {

std::string NoiseState::toString() const {
  switch (noiseType) {
    case (NoiseType::UNINITIALIZED):
      return "NoiseState(uninitialized)";
    case (NoiseType::SET):
      return "NoiseState(" + std::to_string(getValue()) + ", " +
             std::to_string(degree) + ") ";
    default:
      llvm_unreachable("Unknown noise type");
      return "";
  }
}

NoiseState NoiseState::operator+(const NoiseState &rhs) const {
  assert(isKnown() && rhs.isKnown());

  auto degree = std::max(getDegree(), rhs.getDegree());

  // give sum of two log value
  double log2a = getValue();
  double log2b = rhs.getValue();
  // make a >= b
  if (log2b > log2a) {
    std::swap(log2a, log2b);
  }
  // if a >>> b, do not call std::exp2 to avoid overflow
  if (log2b == NEGATIVE_INFINITY || log2a - log2b > 512.0) {
    return NoiseState(NoiseType::SET, log2a, degree);
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
  return NoiseState(NoiseType::SET, log2Sum, degree);
}

NoiseState NoiseState::operator*(const NoiseState &rhs) const {
  assert(isKnown() && rhs.isKnown());

  auto degree = std::max(getDegree(), rhs.getDegree());

  // give product of two log value
  auto log2a = getValue();
  auto log2b = rhs.getValue();
  // log2(a * b) = log2(a) + log2(b)
  // this works for negative infinity
  return NoiseState(NoiseType::SET, log2a + log2b, degree);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const NoiseState &noise) {
  return os << noise.toString();
}

Diagnostic &operator<<(Diagnostic &diagnostic, const NoiseState &noise) {
  return diagnostic << noise.toString();
}

}  // namespace heir
}  // namespace mlir
