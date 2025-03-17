#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEWITHDEGREE_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEWITHDEGREE_H_

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {

class NoiseWithDegree {
 public:
  enum NoiseType {
    // A min value for the lattice, discarable when joined with anything else.
    UNINITIALIZED,
    // A known value for the lattice, when noise can be inferred.
    SET,
  };

  static NoiseWithDegree uninitialized() {
    return NoiseWithDegree(NoiseType::UNINITIALIZED, std::nullopt);
  }
  static NoiseWithDegree of(double value, int degree) {
    return NoiseWithDegree(NoiseType::SET, value, degree);
  }

  /// The default constructor must be equivalent to the "entry state" of the
  /// lattice, i.e., an uninitialized noise.
  NoiseWithDegree(NoiseType noiseType = NoiseType::UNINITIALIZED,
                  std::optional<double> value = std::nullopt, int degree = 0)
      : noiseType(noiseType), value(value), degree(degree) {}

  bool isKnown() const { return noiseType == NoiseType::SET; }

  bool isInitialized() const { return noiseType != NoiseType::UNINITIALIZED; }

  const double &getValue() const {
    assert(isKnown());
    return *value;
  }

  int getDegree() const {
    assert(isKnown());
    return degree;
  }

  bool operator==(const NoiseWithDegree &rhs) const {
    return noiseType == rhs.noiseType && value == rhs.value &&
           degree == rhs.degree;
  }

  static NoiseWithDegree join(const NoiseWithDegree &lhs,
                              const NoiseWithDegree &rhs) {
    // Uninitialized noises correspond to values that are not secret,
    // which may be the inputs to an encryption operation.
    if (lhs.noiseType == NoiseType::UNINITIALIZED) {
      return rhs;
    }
    if (rhs.noiseType == NoiseType::UNINITIALIZED) {
      return lhs;
    }

    assert(lhs.noiseType == NoiseType::SET && rhs.noiseType == NoiseType::SET);
    return NoiseWithDegree::of(std::max(lhs.getValue(), rhs.getValue()),
                               std::max(lhs.getDegree(), rhs.getDegree()));
  }

  void print(llvm::raw_ostream &os) const { os << value; }

  std::string toString() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const NoiseWithDegree &noise);

  friend Diagnostic &operator<<(Diagnostic &diagnostic,
                                const NoiseWithDegree &noise);

 private:
  NoiseType noiseType;
  // notice that when level becomes large (e.g. 17), the max Q could be like
  // 3523 bits and could not be represented in double.
  std::optional<double> value;
  int degree;
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_NOISEWITHDEGREE_H_
