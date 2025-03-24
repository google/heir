#include "lib/Analysis/NoiseAnalysis/NoiseWithDegree.h"

#include <cmath>
#include <string>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {

std::string NoiseWithDegree::toString() const {
  switch (noiseType) {
    case (NoiseType::UNINITIALIZED):
      return "NoiseWithDegree(uninitialized)";
    case (NoiseType::SET):
      return "NoiseWithDegree(" + std::to_string(log2(getValue())) + ", " +
             std::to_string(degree) + ") ";
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const NoiseWithDegree &noise) {
  return os << noise.toString();
}

Diagnostic &operator<<(Diagnostic &diagnostic, const NoiseWithDegree &noise) {
  return diagnostic << noise.toString();
}

}  // namespace heir
}  // namespace mlir
