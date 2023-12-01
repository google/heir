#include "include/Analysis/NoisePropagation/Variance.h"

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {

std::string Variance::toString() const {
  switch (varianceType) {
    case (VarianceType::UNINITIALIZED):
      return "Variance(uninitialized)";
    case (VarianceType::UNBOUNDED):
      return "Variance(unbounded)";
    case (VarianceType::SET):
      return "Variance(" + std::to_string(getValue()) + ")";
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Variance &variance) {
  return os << variance.toString();
}

Diagnostic &operator<<(Diagnostic &diagnostic, const Variance &variance) {
  return diagnostic << variance.toString();
}

}  // namespace heir
}  // namespace mlir
