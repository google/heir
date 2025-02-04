#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISE_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISE_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <optional>

#include "lib/Analysis/NoiseAnalysis/Params.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

// is worst case or not
template <bool W = false>
class Noise {
 public:
  enum NoiseType {
    // A min value for the lattice, discarable when joined with anything else.
    UNINITIALIZED,
    // A known value for the lattice, when noise can be inferred.
    SET,
  };

  static Noise uninitialized() {
    return Noise(NoiseType::UNINITIALIZED, std::nullopt);
  }
  static Noise of(double value) { return Noise(NoiseType::SET, value); }

  /// Create an integer value range lattice value.
  /// The default constructor must be equivalent to the "entry state" of the
  /// lattice, i.e., an uninitialized noise.
  Noise(NoiseType noiseType = NoiseType::UNINITIALIZED,
        std::optional<double> value = std::nullopt)
      : noiseType(noiseType), value(value) {}

  bool isKnown() const { return noiseType == NoiseType::SET; }

  bool isInitialized() const { return noiseType != NoiseType::UNINITIALIZED; }

  const double &getValue() const {
    assert(isKnown());
    return *value;
  }

  bool operator==(const Noise &rhs) const {
    return noiseType == rhs.noiseType && value == rhs.value;
  }

  static Noise join(const Noise &lhs, const Noise &rhs) {
    // Uninitialized noises correspond to values that are not secret,
    // which may be the inputs to an encryption operation.
    if (lhs.noiseType == NoiseType::UNINITIALIZED) {
      return rhs;
    }
    if (rhs.noiseType == NoiseType::UNINITIALIZED) {
      return lhs;
    }

    assert(lhs.noiseType == NoiseType::SET && rhs.noiseType == NoiseType::SET);
    return Noise::of(std::max(lhs.getValue(), rhs.getValue()));
  }

  static double getExpansionFactor(const LocalParam &param);
  static double getBoundErr(const LocalParam &param);
  static double getBoundKey(const LocalParam &param);

  static Noise evalConstant(const LocalParam &param);
  // std0: std error of e distribution
  // assumed UNIFORM_TENARY secret distribution
  static Noise evalEncryptPk(const LocalParam &param);
  static Noise evalAdd(const Noise &lhs, const Noise &rhs);
  static Noise evalMultNoRelin(const LocalParam &resultParam, const Noise &lhs,
                               const Noise &rhs);
  // l: number of digit
  // beta: base
  static Noise evalRelinearizeBV(const LocalParam &inputParam,
                                 const Noise &input);
  static Noise evalRelinearizeHYBRID(const LocalParam &inputParam,
                                     const Noise &input);
  static Noise evalRelinearize(const LocalParam &inputParam,
                               const Noise &input);
  static Noise evalModReduce(const LocalParam &inputParam, const Noise &input);
  static Noise evalRotate(const LocalParam &inputParam, const Noise &input);

  std::string toBound(const LocalParam &param) const;

  void print(llvm::raw_ostream &os) const { os << value; }

  std::string toString() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Noise &noise);

  friend Diagnostic &operator<<(Diagnostic &diagnostic, const Noise &noise);

 private:
  NoiseType noiseType;
  std::optional<double> value;
};

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_BGV_NOISE_H_
