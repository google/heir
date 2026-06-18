#include "lib/Utils/Polynomial/NTT.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "absl/numeric/int128.h"  // from @com_google_absl
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

inline uint32_t log2_32(uint32_t n) {
  uint32_t res = 0;
  while (n > 1) {
    res++;
    n >>= 1;
  }
  return res;
}

uint32_t reverseBits(uint32_t n, uint32_t bitWidth) {
  uint32_t rev = 0;
  for (uint32_t i = 0; i < bitWidth; ++i) {
    rev = (rev << 1) | (n & 1);
    n >>= 1;
  }
  return rev;
}

inline uint64_t modularMultiplication64(uint64_t a, uint64_t b,
                                        uint64_t modulus) {
  absl::uint128 prod = absl::uint128(a) * absl::uint128(b);
  return static_cast<uint64_t>(prod % modulus);
}

inline uint64_t modularAddition64(uint64_t u, uint64_t v, uint64_t modulus) {
  assert(u < modulus && v < modulus);
  return (u >= modulus - v) ? (u - (modulus - v)) : (u + v);
}

inline uint64_t modularSubtraction64(uint64_t u, uint64_t v, uint64_t modulus) {
  assert(u < modulus && v < modulus);
  return (u < v) ? (modulus - (v - u)) : (u - v);
}

void nttInPlace(std::vector<uint64_t>& coeffs, uint64_t modulus,
                uint64_t rootOfUnity) {
  uint32_t n = coeffs.size();
  if (n <= 1) return;
  assert((n & (n - 1)) == 0 && "n must be a power of 2");

  std::vector<uint64_t> rootOfUnityTable(n, 0);
  uint64_t x = 1;
  for (uint32_t i = 0; i < n; ++i) {
    rootOfUnityTable[i] = x;
    x = modularMultiplication64(x, rootOfUnity, modulus);
  }

  for (uint32_t i = 0; i < n; ++i) {
    coeffs[i] =
        modularMultiplication64(coeffs[i], rootOfUnityTable[i], modulus);
  }

  uint32_t span = (n >> 1);
  for (; span >= 1; span >>= 1) {
    uint32_t numGroups = n / (2 * span);
    for (uint32_t j = 0; j < span; ++j) {
      uint64_t omega = rootOfUnityTable[2 * j * numGroups];
      for (uint32_t group = 0; group < numGroups; ++group) {
        uint32_t i = (group * 2 * span) + j;
        uint64_t u = coeffs[i];
        uint64_t v = coeffs[i + span];
        uint64_t uPrime = modularAddition64(u, v, modulus);
        coeffs[i] = uPrime;
        uint64_t vSub = modularSubtraction64(u, v, modulus);
        coeffs[i + span] = modularMultiplication64(vSub, omega, modulus);
      }
    }
  }
}

void inttInPlace(std::vector<uint64_t>& coeffs, uint64_t modulus,
                 uint64_t rootOfUnity) {
  uint32_t n = coeffs.size();
  if (n <= 1) return;
  assert((n & (n - 1)) == 0 && "n must be a power of 2");

  llvm::APInt rootOfUnityAp(64, rootOfUnity);
  llvm::APInt modulusAp(64, modulus);
  llvm::APInt rootOfUnityInverseAp =
      multiplicativeInverse(rootOfUnityAp, modulusAp);
  uint64_t rootOfUnityInverse = rootOfUnityInverseAp.getZExtValue();

  std::vector<uint64_t> rootOfUnityInverseTable(n, 0);
  uint64_t x = 1;
  for (uint32_t i = 0; i < n; ++i) {
    rootOfUnityInverseTable[i] = x;
    x = modularMultiplication64(x, rootOfUnityInverse, modulus);
  }

  uint32_t span = 1;
  for (; span < n; span <<= 1) {
    uint32_t numGroups = n / (2 * span);
    for (uint32_t j = 0; j < span; ++j) {
      uint64_t omegaInv = rootOfUnityInverseTable[2 * j * numGroups];
      for (uint32_t group = 0; group < numGroups; ++group) {
        uint32_t i = (group * 2 * span) + j;
        uint64_t u = coeffs[i];
        uint64_t v = coeffs[i + span];
        uint64_t vScaled = modularMultiplication64(v, omegaInv, modulus);
        coeffs[i] = modularAddition64(u, vScaled, modulus);
        coeffs[i + span] = modularSubtraction64(u, vScaled, modulus);
      }
    }
  }

  llvm::APInt nAp(64, n);
  llvm::APInt nInvAp = multiplicativeInverse(nAp, modulusAp);
  uint64_t nInv = nInvAp.getZExtValue();

  for (uint32_t i = 0; i < n; ++i) {
    uint64_t factor =
        modularMultiplication64(rootOfUnityInverseTable[i], nInv, modulus);
    coeffs[i] = modularMultiplication64(coeffs[i], factor, modulus);
  }
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
