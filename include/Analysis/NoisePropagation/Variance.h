#ifndef INCLUDE_ANALYSIS_NOISEPROPAGATION_VARIANCE_H_
#define INCLUDE_ANALYSIS_NOISEPROPAGATION_VARIANCE_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {

/// A class representing an optional variance of a noise distribution.
class Variance {
 public:
  static Variance unknown() { return Variance(); }

  /// Create an integer value range lattice value.
  Variance(std::optional<int64_t> value = std::nullopt) : value(value) {}

  bool isKnown() const { return value.has_value(); }

  const int64_t &getValue() const {
    assert(isKnown());
    return *value;
  }

  bool operator==(const Variance &rhs) const { return value == rhs.value; }

  /// This method represents how to choose a noise from one of two possible
  /// branches, when either could be possible. In the case of FHE, we must
  /// assume the worse case, so take the max.
  static Variance join(const Variance &lhs, const Variance &rhs) {
    if (!lhs.isKnown()) return rhs;
    if (!rhs.isKnown()) return lhs;
    return Variance{std::max(lhs.getValue(), rhs.getValue())};
  }

  void print(llvm::raw_ostream &os) const { os << value; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Variance &variance);

  friend Diagnostic &operator<<(Diagnostic &diagnostic,
                                const Variance &variance);

 private:
  std::optional<int64_t> value;
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEPROPAGATION_VARIANCE_H_
