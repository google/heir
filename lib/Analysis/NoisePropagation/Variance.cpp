#include "include/Analysis/NoisePropagation/Variance.h"

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Variance &variance) {
  if (!variance.isKnown()) return os << "unknown";
  return os << variance.getValue();
}

}  // namespace heir
}  // namespace mlir
