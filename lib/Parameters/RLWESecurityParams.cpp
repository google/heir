#include "lib/Parameters/RLWESecurityParams.h"

#include <cassert>

namespace mlir {
namespace heir {

// "Security Guidelines for Implementing Homomorphic Encryption"
// https://ia.cr/2024/463

// 128-bit classic security for uniform ternary secret distribution
struct RLWESecurityParam rlweSecurityParam128BitClassic[] = {
    {1024, 26},   {2048, 53},   {4096, 106},   {8192, 214},
    {16384, 430}, {32768, 868}, {65536, 1747}, {131072, 3523}};

int computeRingDim(int logPQ, int minRingDim) {
  for (auto& param : rlweSecurityParam128BitClassic) {
    if (param.ringDim < minRingDim) {
      continue;
    }
    if (param.logMaxQ >= logPQ) {
      return param.ringDim;
    }
  }
  assert(false && "Failed to find ring dimension, logTotalPQ too large");
  return 0;
}

}  // namespace heir
}  // namespace mlir
