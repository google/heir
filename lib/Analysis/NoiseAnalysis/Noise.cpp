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
      return "NoiseState(" + std::to_string(log2(getValue())) + ") ";
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const NoiseState &noise) {
  return os << noise.toString();
}

Diagnostic &operator<<(Diagnostic &diagnostic, const NoiseState &noise) {
  return diagnostic << noise.toString();
}

}  // namespace heir
}  // namespace mlir
