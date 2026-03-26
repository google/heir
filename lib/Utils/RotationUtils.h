#ifndef LIB_UTILS_ROTATIONUTILS_H_
#define LIB_UTILS_ROTATIONUTILS_H_

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
  int64_t maxRatio = 1LL << logMaxRatio;
  for (int64_t n1 = 1; n1 < slots; n1 <<= 1) {
    llvm::DenseSet<int64_t> rotN1Set, rotN2Set;
    for (auto rot : diags) {
      int64_t r = normalizeRotation(rot, slots);
      rotN1Set.insert(((r / n1) * n1) & (slots - 1));
      rotN2Set.insert(r & (n1 - 1));
    }
    int64_t nbN1 = static_cast<int64_t>(rotN1Set.size()) - 1;
    int64_t nbN2 = static_cast<int64_t>(rotN2Set.size()) - 1;
    if (nbN1 > 0) {
      if (nbN2 == maxRatio * nbN1) return n1;
      if (nbN2 > maxRatio * nbN1) return n1 / 2;
    }
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
    int64_t giant = ((r / n1) * n1) & (slots - 1);
    int64_t baby = r & (n1 - 1);
    if (giant != 0) result.insert(giant);
    if (baby != 0) result.insert(baby);
  }
  return result;
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ROTATIONUTILS_H_
