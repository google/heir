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

enum VarianceType {
  // A min value for the lattice, discarable when joined with anything else.
  UNINITIALIZED,
  // A known value for the lattice, when noise can be inferred.
  SET,
  // A max value for the lattice, when noise cannot be inferred and a bootstrap
  // must be forced.
  UNBOUNDED
};

/// A class representing an optional variance of a noise distribution.
class Variance {
 public:
  static Variance uninitialized() {
    return Variance(VarianceType::UNINITIALIZED, std::nullopt);
  }
  static Variance unbounded() {
    return Variance(VarianceType::UNBOUNDED, std::nullopt);
  }
  static Variance of(int64_t value) {
    return Variance(VarianceType::SET, value);
  }

  /// Create an integer value range lattice value.
  /// The default constructor must be equivalent to the "entry state" of the
  /// lattice, i.e., an uninitialized noise variance.
  Variance(VarianceType varianceType = VarianceType::UNINITIALIZED,
           std::optional<int64_t> value = std::nullopt)
      : varianceType(varianceType), value(value) {}

  bool isKnown() const { return varianceType == VarianceType::SET; }

  bool isInitialized() const {
    return varianceType != VarianceType::UNINITIALIZED;
  }

  bool isBounded() const { return varianceType != VarianceType::UNBOUNDED; }

  const int64_t &getValue() const {
    assert(isKnown());
    return *value;
  }

  bool operator==(const Variance &rhs) const {
    return varianceType == rhs.varianceType && value == rhs.value;
  }

  static Variance join(const Variance &lhs, const Variance &rhs) {
    // Uninitialized variances correspond to values that are not secret,
    // which may be the inputs to an encryption operation.
    if (lhs.varianceType == VarianceType::UNINITIALIZED) {
      return rhs;
    }
    if (rhs.varianceType == VarianceType::UNINITIALIZED) {
      return lhs;
    }

    // Unbounded represents a pessimistic worst case, and so it must be
    // preserved no matter the other operand.
    if (lhs.varianceType == VarianceType::UNBOUNDED) {
      return lhs;
    }
    if (rhs.varianceType == VarianceType::UNBOUNDED) {
      return rhs;
    }

    assert(lhs.varianceType == VarianceType::SET &&
           rhs.varianceType == VarianceType::SET);
    return Variance::of(std::max(lhs.getValue(), rhs.getValue()));
  }

  void print(llvm::raw_ostream &os) const { os << value; }

  std::string toString() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Variance &variance);

  friend Diagnostic &operator<<(Diagnostic &diagnostic,
                                const Variance &variance);

 private:
  VarianceType varianceType;
  std::optional<int64_t> value;
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEPROPAGATION_VARIANCE_H_
