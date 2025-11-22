#ifndef LIB_ANALYSIS_NOISEANALYSIS_NOISE_H_
#define LIB_ANALYSIS_NOISEANALYSIS_NOISE_H_

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>

#include "lib/Utils/LogArithmetic.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {

// An enum some noise models can use to tweak their behavior for different
// noise modeling regimes. Worst case uses worst-case bounds on noise growth,
// while average case uses statistical bounds, such as a range of standard
// deviations away from the expected noise grwoth.
enum NoiseModelVariant {
  AVERAGE_CASE,
  WORST_CASE,
};

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
  /// value in normal scale
  static NoiseState of(double value) {
    return NoiseState::of(value, /*degree*/ 0);
  }
  /// value in normal scale
  static NoiseState of(double value, int degree) {
    return NoiseState(NoiseType::SET, Log2Arithmetic::of(value), degree);
  }

  /// Create an integer value range lattice value.
  /// The default constructor must be equivalent to the "entry state" of the
  /// lattice, i.e., an uninitialized noise.
  NoiseState(NoiseType noiseType = NoiseType::UNINITIALIZED,
             std::optional<Log2Arithmetic> value = std::nullopt, int degree = 0)
      : noiseType(noiseType), value(value), degree(degree) {}

  bool isKnown() const { return noiseType == NoiseType::SET; }

  bool isInitialized() const { return noiseType != NoiseType::UNINITIALIZED; }

  // this returns log2(e) instead of e, use with caution
  double getValue() const {
    assert(isKnown());
    return value->getLog2Value();
  }

  Log2Arithmetic getLog2Arithmetic() const {
    assert(isKnown());
    return *value;
  }

  int getDegree() const {
    assert(isKnown());
    return degree;
  }

  NoiseState withNewDegree(int degree) {
    return NoiseState(noiseType, value, degree);
  }

  bool operator==(const NoiseState& rhs) const {
    return noiseType == rhs.noiseType && value == rhs.value &&
           degree == rhs.degree;
  }

  bool operator!=(const NoiseState& rhs) const { return !(*this == rhs); }

  // Although NoiseState stores log(e), we expose the operation
  // as if we are doing e1 + e2 using log(e1) and log(e2)
  NoiseState operator+(const NoiseState& rhs) const;

  NoiseState& operator+=(const NoiseState& rhs) {
    assert(isKnown());
    return *this = *this + rhs;
  }

  // for int/double
  template <typename T>
  NoiseState operator+(const T& rhs) const {
    assert(isKnown());
    return *this + NoiseState::of(rhs);
  }

  template <typename T>
  NoiseState& operator+=(const T& rhs) {
    assert(isKnown());
    return *this += NoiseState::of(rhs);
  }

  // Although NoiseState stores log(e), we expose the operation
  // as if we are doing e1 * e2 using log(e1) and log(e2)
  NoiseState operator*(const NoiseState& rhs) const;

  NoiseState& operator*=(const NoiseState& rhs) {
    assert(isKnown());
    return *this = *this * rhs;
  }

  // for int/double
  template <typename T>
  NoiseState operator*(const T& rhs) const {
    assert(isKnown());
    return *this * NoiseState::of(rhs);
  }

  template <typename T>
  NoiseState& operator*=(const T& rhs) {
    assert(isKnown());
    return *this *= NoiseState::of(rhs);
  }

  static NoiseState join(const NoiseState& lhs, const NoiseState& rhs) {
    // Uninitialized noises correspond to values that are not secret,
    // which may be the inputs to an encryption operation.
    if (lhs.noiseType == NoiseType::UNINITIALIZED) {
      return rhs;
    }
    if (rhs.noiseType == NoiseType::UNINITIALIZED) {
      return lhs;
    }

    assert(lhs.noiseType == NoiseType::SET && rhs.noiseType == NoiseType::SET);
    return NoiseState(
        NoiseType::SET,
        std::max(lhs.getLog2Arithmetic(), rhs.getLog2Arithmetic()),
        std::max(lhs.getDegree(), rhs.getDegree()));
  }

  void print(llvm::raw_ostream& os) const { os << toString(); }

  std::string toString() const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const NoiseState& noise);

  friend Diagnostic& operator<<(Diagnostic& diagnostic,
                                const NoiseState& noise);

 private:
  NoiseType noiseType;
  // notice that when level becomes large (e.g. 17), the max Q could be like
  // 3523 bits and could not be represented in double.
  // To mitigate such problem we store log2(Noise) as only the order of
  // magnititude is important
  std::optional<Log2Arithmetic> value;
  // for some analysis a degree is tracked
  int degree;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_NOISEANALYSIS_NOISE_H_
