#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_NOISE_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_NOISE_H_

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {

// This class could be shared among all noise models that tracks the noise by a
// single value. Noise model could have different interpretation of the value.
// In BGV/BFV world, most noise model just use a single value, either as bound
// or as variance.
class NoiseState {
 public:
  enum NoiseType {
    // A min value for the lattice, discarable when joined with anything else.
    UNINITIALIZED,
    // A known value for the lattice, when noise can be inferred.
    SET,
  };

  static NoiseState uninitialized() {
    return NoiseState(NoiseType::UNINITIALIZED, std::nullopt);
  }
  static NoiseState of(double value) {
    return NoiseState(NoiseType::SET, value);
  }

  /// Create an integer value range lattice value.
  /// The default constructor must be equivalent to the "entry state" of the
  /// lattice, i.e., an uninitialized noise.
  NoiseState(NoiseType noiseType = NoiseType::UNINITIALIZED,
             std::optional<double> value = std::nullopt)
      : noiseType(noiseType), value(value) {}

  bool isKnown() const { return noiseType == NoiseType::SET; }

  bool isInitialized() const { return noiseType != NoiseType::UNINITIALIZED; }

  const double &getValue() const {
    assert(isKnown());
    return *value;
  }

  bool operator==(const NoiseState &rhs) const {
    return noiseType == rhs.noiseType && value == rhs.value;
  }

  static NoiseState join(const NoiseState &lhs, const NoiseState &rhs) {
    // Uninitialized noises correspond to values that are not secret,
    // which may be the inputs to an encryption operation.
    if (lhs.noiseType == NoiseType::UNINITIALIZED) {
      return rhs;
    }
    if (rhs.noiseType == NoiseType::UNINITIALIZED) {
      return lhs;
    }

    assert(lhs.noiseType == NoiseType::SET && rhs.noiseType == NoiseType::SET);
    return NoiseState::of(std::max(lhs.getValue(), rhs.getValue()));
  }

  void print(llvm::raw_ostream &os) const { os << value; }

  std::string toString() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const NoiseState &noise);

  friend Diagnostic &operator<<(Diagnostic &diagnostic,
                                const NoiseState &noise);

 private:
  NoiseType noiseType;
  // notice that when level becomes large (e.g. 17), the max Q could be like
  // 3523 bits and could not be represented in double.
  std::optional<double> value;
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_NOISE_H_
