#ifndef LIB_UTILS_ROTATIONUTILS_H_
#define LIB_UTILS_ROTATIONUTILS_H_

#include <cmath>
#include <cstdint>

#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// Normalize a rotation index into [0, slots).
inline int64_t normalizeRotation(int64_t rot, int64_t slots) {
  return ((rot % slots) + slots) % slots;
}

/// Returns the best baby-step size N1 for BSGS
/// given diagonal indices, slot count, and log2 of the target baby/giant ratio.
/// Mirrors Lattigo's lintrans.FindBestBSGSRatio.
inline int64_t findBestBSGSRatio(llvm::ArrayRef<int32_t> diags, int64_t slots,
                                 int64_t logMaxRatio) {
  double maxRatio = static_cast<double>(1LL << logMaxRatio);
  for (int64_t n1 = 1; n1 < slots; n1 <<= 1) {
    llvm::DenseSet<int64_t> rotN1Set, rotN2Set;
    for (auto rot : diags) {
      int64_t r = normalizeRotation(rot, slots);
      rotN1Set.insert(normalizeRotation((r / n1) * n1, slots));
      rotN2Set.insert(r % n1);
    }
    double nbN1 = static_cast<double>(rotN1Set.size()) - 1;
    double nbN2 = static_cast<double>(rotN2Set.size()) - 1;
    // Float division without a nonzero guard, as Lattigo does it.
    double ratio = nbN2 / nbN1;
    if (ratio == maxRatio) return n1;
    if (ratio > maxRatio) return n1 / 2;
  }
  return 1;
}

/// Returns all non-zero rotation indices needed by a linear transform
/// with the given diagonal indices, slot count, and BSGS ratio.
/// Mirrors Lattigo's lintrans.GaloisElements().
inline llvm::DenseSet<int64_t> lintransRotationIndices(
    llvm::ArrayRef<int32_t> diags, int64_t slots, int64_t logBSGS) {
  llvm::DenseSet<int64_t> result;
  int64_t n1 = (logBSGS < 0) ? slots : findBestBSGSRatio(diags, slots, logBSGS);
  for (auto rot : diags) {
    int64_t r = normalizeRotation(rot, slots);
    int64_t giant = normalizeRotation((r / n1) * n1, slots);
    int64_t baby = r % n1;
    if (giant != 0) result.insert(giant);
    if (baby != 0) result.insert(baby);
  }
  return result;
}

/// Returns the ciphertext rotation indices needed by a rotate-and-reduce op.
///
/// Without plaintexts: binary span-doubling shifts.
/// With plaintexts: BSGS, using ceil(sqrt(steps)) as the split.
inline llvm::DenseSet<int64_t> rotateAndReduceRotationIndices(
    int64_t period, int64_t steps, bool hasPlaintexts) {
  llvm::DenseSet<int64_t> result;
  if (!hasPlaintexts) {
    // Matches implementRotateAndReduceAccumulation
    int64_t offset = 0;
    for (int64_t shiftSize = 1; shiftSize <= steps; shiftSize *= 2) {
      if (steps & shiftSize) {
        if (offset != 0) result.insert(offset * period);
        offset += shiftSize;
        if (offset == steps) break;
      }
      result.insert(shiftSize * period);
    }
    return result;
  }

  // Matches implementBabyStepGiantStep
  int64_t numBabySteps = static_cast<int64_t>(std::ceil(std::sqrt(steps)));
  int64_t giantStepSize = numBabySteps;
  int64_t numGiantSteps = (steps + numBabySteps - 1) / numBabySteps;

  // Baby step ciphertext rotations
  for (int64_t i = 1; i < numBabySteps; ++i) {
    result.insert(period * i);
  }
  // Giant step ciphertext rotations
  for (int64_t j = 1; j < numGiantSteps; ++j) {
    result.insert(period * j * giantStepSize);
  }
  return result;
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ROTATIONUTILS_H_
