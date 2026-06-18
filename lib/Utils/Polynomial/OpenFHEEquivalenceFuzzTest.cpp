#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

// IWYU pragma: begin_keep
#include "src/core/include/lattice/hal/lat-backend.h"    // from @openfhe
#include "src/core/include/math/hal/nativeintbackend.h"  // from @openfhe
#include "src/core/include/utils/inttypes.h"             // from @openfhe
#include "src/pke/include/openfhe.h"                     // from @openfhe
// IWYU pragma: end_keep

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Polynomial/NTT.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

using ::lbcrypto::ILNativeParams;
using ::lbcrypto::NativeInteger;
using ::lbcrypto::NativePoly;

// Pre-generated static table of diverse, mathematically valid (N, q) pairs.
// N must be a power of 2.
// q must be a prime such that 2N divides q - 1.
struct NTTParameters {
  uint32_t n;
  uint64_t q;
};

const std::vector<NTTParameters> kNTTParameters = {
    {8, 65537},
    {64, 65537},
    {512, 65537},
    {1024, 65537},
    {2048, 65537},
    {4096, 65537},
    {8, 12289},
    {64, 12289},
    {1024, 12289},
    {2048, 12289},
    // Primes >= 2^63 to trigger the 128-bit safeWidth path.
    {8, 9223372036854776257ULL},  // Supports N up to 32
    {32, 9223372036854776257ULL},
    {8, 9223372036855103489ULL},  // Supports N up to 32768
    {64, 9223372036855103489ULL},
    {1024, 9223372036855103489ULL},
    {4096, 9223372036855103489ULL},
};

void ForwardNTTAndInverseINTTEquivalence(NTTParameters params,
                                         const std::vector<uint64_t>& coeffs) {
  uint32_t n = params.n;
  uint64_t q = params.q;

  if (coeffs.empty()) return;

  // Derive coefficients of length N, bounded by [0, q-1].
  std::vector<uint64_t> rawCoeffs(n, 0);
  for (size_t i = 0; i < n; ++i) {
    rawCoeffs[i] = coeffs[i % coeffs.size()] % q;
  }

  // Derive root of unity using APInt (required by MathUtils)
  unsigned safeWidth = (q < (1ULL << 63)) ? 64 : 128;
  llvm::APInt qAp(safeWidth, q);
  std::optional<llvm::APInt> rootOfUnityOpt = findPrimitive2nthRoot(qAp, n);
  ASSERT_TRUE(rootOfUnityOpt.has_value())
      << "Failed to find primitive root for N=" << n << ", q=" << q;
  uint64_t root = rootOfUnityOpt->getZExtValue();

  std::vector<uint64_t> heirCoeffs = rawCoeffs;

  // OpenFHE's NativeInteger is fixed to 64-bit in this build configuration.
  // When q >= 2^63, intermediate additions (a + b) in the butterfly operations
  // can exceed 2^64 - 1, causing incorrect results in OpenFHE. Our in-memory
  // NTT avoids this by dynamically switching to safeWidth = 128. Therefore, for
  // q >= 2^63, we bypass the OpenFHE equivalence check and rely purely on our
  // rigorous Roundtrip Check.
  if (q < (1ULL << 63)) {
    NativeInteger nativeModulus(q);
    NativeInteger nativeRoot(root);
    uint32_t m = 2 * n;
    auto nativeParams =
        std::make_shared<ILNativeParams>(m, nativeModulus, nativeRoot);

    NativePoly openfhePoly(nativeParams, Format::COEFFICIENT);
    NativePoly::Vector openfheVec(n, nativeModulus);
    for (uint32_t i = 0; i < n; ++i) {
      openfheVec[i] = NativeInteger(rawCoeffs[i]);
    }
    openfhePoly.SetValues(std::move(openfheVec), Format::COEFFICIENT);

    // Verify initial exact match
    for (uint32_t i = 0; i < n; ++i) {
      EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i])
          << "Initial exact match mismatch at coefficient index " << i
          << " for N=" << n << ", q=" << q;
    }

    // Forward NTT Equivalence
    openfhePoly.SwitchFormat();
    nttInPlace(heirCoeffs, q, root);

    for (uint32_t i = 0; i < n; ++i) {
      EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i])
          << "Mismatch at evaluation index " << i << " for N=" << n
          << ", q=" << q;
    }

    // Inverse INTT Equivalence
    openfhePoly.SwitchFormat();
    inttInPlace(heirCoeffs, q, root);

    for (uint32_t i = 0; i < n; ++i) {
      EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i])
          << "Mismatch at coefficient index " << i << " for N=" << n
          << ", q=" << q;
    }
  } else {
    // For large moduli, just execute our NTT and INTT to verify roundtrip.
    nttInPlace(heirCoeffs, q, root);
    inttInPlace(heirCoeffs, q, root);
  }

  // Roundtrip Check
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(heirCoeffs[i], rawCoeffs[i])
        << "Roundtrip mismatch at coefficient index " << i << " for N=" << n
        << ", q=" << q;
  }
}

FUZZ_TEST(OpenFHEEquivalenceFuzzTest, ForwardNTTAndInverseINTTEquivalence)
    .WithDomains(fuzztest::ElementOf<NTTParameters>(kNTTParameters),
                 fuzztest::VectorOf(fuzztest::Arbitrary<uint64_t>())
                     .WithMinSize(1)
                     .WithMaxSize(4096));

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
