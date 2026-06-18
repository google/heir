#include <cstdint>
#include <memory>
#include <optional>
#include <random>
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
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {
namespace {

using ::lbcrypto::ILNativeParams;
using ::lbcrypto::NativeInteger;
using ::lbcrypto::NativePoly;

TEST(OpenFHEEquivalenceTest, ForwardNTTAndInverseINTT_N8) {
  uint32_t n = 8;
  uint32_t m = 16;
  uint64_t q = 65537;

  llvm::APInt qAp(64, q);
  std::optional<llvm::APInt> rootOfUnityOpt = findPrimitive2nthRoot(qAp, n);
  ASSERT_TRUE(rootOfUnityOpt.has_value());
  uint64_t root = rootOfUnityOpt->getZExtValue();

  NativeInteger nativeModulus(q);
  NativeInteger nativeRoot(root);
  auto params = std::make_shared<ILNativeParams>(m, nativeModulus, nativeRoot);

  std::vector<int64_t> rawCoeffs = {431,  3414, 1234, 7845,
                                    2145, 7415, 5471, 8452};

  NativePoly openfhePoly(params, Format::COEFFICIENT);
  openfhePoly = {431, 3414, 1234, 7845, 2145, 7415, 5471, 8452};

  IntPolynomial heirPoly = IntPolynomial::fromCoefficients(rawCoeffs);
  std::vector<uint64_t> heirCoeffs(n, 0);
  for (const auto& term : heirPoly.getTerms()) {
    uint32_t exp = term.getExponent().getZExtValue();
    if (exp < n) {
      heirCoeffs[exp] = term.getCoefficient().urem(qAp).getZExtValue();
    }
  }

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i]);
  }

  openfhePoly.SwitchFormat();
  nttInPlace(heirCoeffs, q, root);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i])
        << "Mismatch at evaluation index " << i;
  }

  openfhePoly.SwitchFormat();
  inttInPlace(heirCoeffs, q, root);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i])
        << "Mismatch at coefficient index " << i;
    EXPECT_EQ(heirCoeffs[i], rawCoeffs[i]);
  }
}

TEST(OpenFHEEquivalenceTest, ForwardNTTAndInverseINTT_N64) {
  uint32_t n = 64;
  uint32_t m = 128;
  uint64_t q = 65537;

  llvm::APInt qAp(64, q);
  std::optional<llvm::APInt> rootOfUnityOpt = findPrimitive2nthRoot(qAp, n);
  ASSERT_TRUE(rootOfUnityOpt.has_value());
  uint64_t root = rootOfUnityOpt->getZExtValue();

  NativeInteger nativeModulus(q);
  NativeInteger nativeRoot(root);
  auto params = std::make_shared<ILNativeParams>(m, nativeModulus, nativeRoot);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int64_t> dist(0, q - 1);
  std::vector<int64_t> rawCoeffs(n);
  for (uint32_t i = 0; i < n; ++i) {
    rawCoeffs[i] = dist(rng);
  }

  NativePoly openfhePoly(params, Format::COEFFICIENT);
  NativePoly::Vector openfheVec(n, nativeModulus);
  for (uint32_t i = 0; i < n; ++i) {
    openfheVec[i] = NativeInteger(rawCoeffs[i]);
  }
  openfhePoly.SetValues(std::move(openfheVec), Format::COEFFICIENT);

  IntPolynomial heirPoly = IntPolynomial::fromCoefficients(rawCoeffs);
  std::vector<uint64_t> heirCoeffs(n, 0);
  for (const auto& term : heirPoly.getTerms()) {
    uint32_t exp = term.getExponent().getZExtValue();
    if (exp < n) {
      heirCoeffs[exp] = term.getCoefficient().urem(qAp).getZExtValue();
    }
  }

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i]);
  }

  openfhePoly.SwitchFormat();
  nttInPlace(heirCoeffs, q, root);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i])
        << "Mismatch at evaluation index " << i;
  }

  openfhePoly.SwitchFormat();
  inttInPlace(heirCoeffs, q, root);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(openfhePoly[i].ConvertToInt(), heirCoeffs[i])
        << "Mismatch at coefficient index " << i;
    EXPECT_EQ(heirCoeffs[i], rawCoeffs[i]);
  }
}

TEST(OpenFHEEquivalenceTest, ForwardNTTAndInverseINTT_LargeModulus) {
  uint32_t n = 8;
  // 9223372036854776257 is a prime. 0x80000000000001C1
  uint64_t q = 0x80000000000001C1ULL;
  llvm::APInt qAp(64, q);

  std::optional<llvm::APInt> rootOfUnityOpt = findPrimitive2nthRoot(qAp, n);
  ASSERT_TRUE(rootOfUnityOpt.has_value());
  uint64_t root = rootOfUnityOpt->getZExtValue();

  std::vector<uint64_t> originalCoeffs(n, 0);
  originalCoeffs[0] = q - 1;
  originalCoeffs[1] = q - 2;
  originalCoeffs[2] = q - 3;
  originalCoeffs[3] = 42;

  std::vector<uint64_t> coeffs = originalCoeffs;

  nttInPlace(coeffs, q, root);
  inttInPlace(coeffs, q, root);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(coeffs[i], originalCoeffs[i]) << "Mismatch at index " << i;
  }
}

}  // namespace
}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
