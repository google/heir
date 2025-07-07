#include "lib/Analysis/NoiseAnalysis/Noise.h"

#include <algorithm>
#include <cassert>
#include <string>

#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project

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

  auto value = getLog2Arithmetic() + rhs.getLog2Arithmetic();
  auto degree = std::max(getDegree(), rhs.getDegree());

  return NoiseState(NoiseType::SET, value, degree);
}

NoiseState NoiseState::operator*(const NoiseState &rhs) const {
  assert(isKnown() && rhs.isKnown());

  auto value = getLog2Arithmetic() * rhs.getLog2Arithmetic();
  auto degree = std::max(getDegree(), rhs.getDegree());

  return NoiseState(NoiseType::SET, value, degree);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const NoiseState &noise) {
  return os << noise.toString();
}

Diagnostic &operator<<(Diagnostic &diagnostic, const NoiseState &noise) {
  return diagnostic << noise.toString();
}

}  // namespace heir
}  // namespace mlir
