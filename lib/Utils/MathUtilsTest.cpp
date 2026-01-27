#include <cstdint>
#include <optional>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"               // from @fuzztest
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

TEST(MathUtilsTest, FindPrimitiveRoot7) {
  auto root = findPrimitiveRoot(APInt(64, 7));
  ASSERT_TRUE(root.has_value());
  // Primitive roots of 7 are 3 and 5.
  EXPECT_TRUE(*root == 3 || *root == 5);
}

TEST(MathUtilsTest, FindPrimitiveRoot114689) {
  auto root = findPrimitiveRoot(APInt(64, 114689));
  ASSERT_TRUE(root.has_value());
  // 114689 is prime. Check that root^(q-1) == 1 mod q and root^((q-1)/p) != 1
  // mod q
  APInt q(64, 114689);
  APInt phi = q - 1;
  EXPECT_EQ(modularExponentiation(*root, phi, q), 1);
  auto factors = factorize(phi);
  for (const auto& p : factors) {
    EXPECT_NE(modularExponentiation(*root, phi.udiv(p), q), 1);
  }
}

TEST(MathUtilsTest, FindPrimitiveRootNonPrime) {
  auto root = findPrimitiveRoot(APInt(64, 18));
  ASSERT_FALSE(root.has_value());
}

TEST(MathUtilsTest, FindPrimitive2nthRoot) {
  // q = 114689, n = 1024, 2n = 2048. 114689 - 1 = 114688. 114688 / 2048 = 56.
  auto root = findPrimitive2nthRoot(APInt(64, 114689), 1024);
  ASSERT_TRUE(root.has_value());
  APInt q(64, 114689);
  uint64_t two_n = 2048;
  APInt two_n_ap(64, two_n);
  EXPECT_EQ(modularExponentiation(*root, two_n_ap, q), 1);

  // Check it's PRIMITIVE 2n-th root
  auto factors = factorize(two_n_ap);
  for (const auto& p : factors) {
    EXPECT_NE(modularExponentiation(*root, two_n_ap.udiv(p), q), 1);
  }
}

void Primitive2nthRootProperty(uint64_t q_val, uint64_t n) {
  APInt q(64, q_val);
  if (!isPrime(q)) return;
  if ((q_val - 1) % (2 * n) != 0) return;

  auto root = findPrimitive2nthRoot(q, n);
  ASSERT_TRUE(root.has_value());

  APInt two_n_ap(64, 2 * n);
  // root^(2n) == 1 mod q
  EXPECT_EQ(modularExponentiation(*root, two_n_ap, q), 1);

  // root^(2n/p) != 1 mod q for all prime factors p of 2n
  auto factors = factorize(two_n_ap);
  for (const auto& p : factors) {
    EXPECT_NE(modularExponentiation(*root, two_n_ap.udiv(p), q), 1);
  }
}

// Fuzz test with a set of known NTT-friendly primes and various degrees
std::vector<uint64_t> ntt_primes = {
    65537,      114689,     147457,     163841,     557057,
    638977,     737281,     786433,     1032193,    1179649,
    1769473,    1785857,    2277377,    2424833,    2572289,
    2654209,    2752513,    2768897,    8380417,    2147565569,
    2148155393, 2148384769, 3221225473, 3221241857, 3758161921};

FUZZ_TEST(MathUtilsTest, Primitive2nthRootProperty)
    .WithDomains(fuzztest::ElementOf(ntt_primes),
                 fuzztest::ElementOf({256, 512, 1024, 2048, 4096, 8192}));

}  // namespace
}  // namespace heir
}  // namespace mlir
