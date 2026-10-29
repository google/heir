#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "llvm/include/llvm/ADT/APInt.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_ref {
namespace {

using ::llvm::APInt;

// 32-bit test primes: from bootparam_40.json plus edge primes near 2^31 and
// small primes.
constexpr uint32_t kPrimes32[] = {
    3,           // smallest odd prime
    65537,       // Fermat prime
    30539777,    // bootparam_40.json terminal prime
    485621761,   // bootparam_40.json main prime
    974258177,   // bootparam_40.json main prime
    1081212929,  // bootparam_40.json main prime
    2147352577,  // bootparam_40.json auxiliary prime (near 2^31)
    2147483647,  // 2^31 - 1 (Mersenne prime M31)
};

// 64-bit test primes: from bootparam_40_64bit.json plus edge primes near 2^63
// and small primes.
constexpr uint64_t kPrimes64[] = {
    3ULL,                   // smallest odd prime
    65537ULL,               // Fermat prime
    1099515691009ULL,       // bootparam_40_64bit.json main prime
    1125899908022273ULL,    // bootparam_40_64bit.json main prime
    36028797019488257ULL,   // bootparam_40_64bit.json main prime
    72057594038321153ULL,   // bootparam_40_64bit.json aux prime
    9223372036854775783ULL  // largest prime < 2^63
};

// Oracle for R^(-1) mod q using APInt.
APInt ComputeRInvAPInt(unsigned width, const APInt& q) {
  APInt wide_q = q.zext(width * 2);
  APInt wide_r = APInt::getOneBitSet(width * 2, width);
  APInt r_mod_q = wide_r.urem(wide_q).trunc(width);
  return multiplicativeInverse(r_mod_q, q);
}

// Oracle for Montgomery product using APInt.
template <unsigned W>
typename MontArith<W>::Word MultMontOracleAPInt(typename MontArith<W>::Word a,
                                                typename MontArith<W>::Word b,
                                                typename MontArith<W>::Word q,
                                                const APInt& r_inv) {
  APInt a_ap(2 * W, a);
  APInt b_ap(2 * W, b);
  APInt q_ap(2 * W, q);
  APInt r_inv_ap = r_inv.zext(2 * W);
  APInt prod = (a_ap * b_ap).urem(q_ap);
  APInt res = (prod * r_inv_ap).urem(q_ap);
  return static_cast<typename MontArith<W>::Word>(res.getZExtValue());
}

// -----------------------------------------------------------------------------
// 32-bit tests
// -----------------------------------------------------------------------------

TEST(MontArithTest32, ComputeQInv) {
  for (uint32_t q : kPrimes32) {
    int32_t q_inv = MontArith<32>::ComputeQInv(q);
    // Verified: (q * q_inv) == 1 mod 2^32.
    EXPECT_EQ(static_cast<uint32_t>(q_inv) * q, 1u)
        << "Failed for prime q = " << q;
  }
}

TEST(MontArithTest32, ComputeRAndR2) {
  for (uint32_t q : kPrimes32) {
    uint32_t r = MontArith<32>::ComputeR(q);
    uint64_t expected_r = (1ULL << 32) % q;
    EXPECT_EQ(r, expected_r) << "Failed ComputeR for q = " << q;

    uint32_t r2 = MontArith<32>::ComputeR2(q);
    uint64_t expected_r2 = (static_cast<uint64_t>(r) * r) % q;
    EXPECT_EQ(r2, expected_r2) << "Failed ComputeR2 for q = " << q;
  }
}

TEST(MontArithTest32, AddSubNegateNormalize) {
  for (uint32_t q : kPrimes32) {
    std::vector<uint32_t> test_vals = {0,           1,     2,    q / 2,
                                       (q / 2) + 1, q - 2, q - 1};

    for (uint32_t a : test_vals) {
      // Negate
      uint32_t neg_a = MontArith<32>::Negate(a, q);
      uint32_t exp_neg = (a == 0) ? 0 : q - a;
      EXPECT_EQ(neg_a, exp_neg) << "Negate failed for a=" << a << ", q=" << q;

      // Normalize: output in [-(q-1)/2, (q-1)/2]
      int32_t norm_a = MontArith<32>::Normalize(a, q);
      int32_t exp_norm = (a > (q >> 1))
                             ? static_cast<int32_t>(a) - static_cast<int32_t>(q)
                             : static_cast<int32_t>(a);
      EXPECT_EQ(norm_a, exp_norm)
          << "Normalize failed for a=" << a << ", q=" << q;
      EXPECT_GE(norm_a, -static_cast<int32_t>(q >> 1));
      EXPECT_LE(norm_a, static_cast<int32_t>(q >> 1));

      for (uint32_t b : test_vals) {
        // Add
        uint32_t sum = MontArith<32>::Add(a, b, q);
        uint32_t exp_sum =
            static_cast<uint32_t>((static_cast<uint64_t>(a) + b) % q);
        EXPECT_EQ(sum, exp_sum)
            << "Add failed for a=" << a << ", b=" << b << ", q=" << q;

        // Sub
        uint32_t diff = MontArith<32>::Sub(a, b, q);
        uint32_t exp_diff =
            static_cast<uint32_t>((static_cast<uint64_t>(a) + q - (b % q)) % q);
        EXPECT_EQ(diff, exp_diff)
            << "Sub failed for a=" << a << ", b=" << b << ", q=" << q;
      }
    }
  }
}

TEST(MontArithTest32, ToMontFromMontRoundTrip) {
  for (uint32_t q : kPrimes32) {
    int32_t q_inv = MontArith<32>::ComputeQInv(q);
    uint32_t r2 = MontArith<32>::ComputeR2(q);

    std::vector<uint32_t> test_vals = {0, 1, 2, q / 2, q - 1};
    for (uint32_t a : test_vals) {
      uint32_t a_mont = MontArith<32>::ToMont(a, q);
      uint64_t expected_mont = (static_cast<uint64_t>(a) << 32) % q;
      EXPECT_EQ(a_mont, expected_mont)
          << "ToMont failed for a=" << a << ", q=" << q;

      uint32_t a_mont_via_r2 = MontArith<32>::ToMont(a, r2, q, q_inv);
      EXPECT_EQ(a_mont_via_r2, a_mont);

      uint32_t recovered = MontArith<32>::FromMont(a_mont, q, q_inv);
      EXPECT_EQ(recovered, a)
          << "FromMont roundtrip failed for a=" << a << ", q=" << q;
    }
  }
}

TEST(MontArithTest32, MontReduceLazyRangeAndValues) {
  for (uint32_t q : kPrimes32) {
    int32_t q_inv = MontArith<32>::ComputeQInv(q);
    APInt r_inv = ComputeRInvAPInt(32, APInt(32, q));

    int64_t half_base = 1LL << 31;
    int64_t min_a = -static_cast<int64_t>(q) * half_base;
    int64_t max_a = static_cast<int64_t>(q) * half_base - 1;

    std::vector<int64_t> test_a = {
        min_a, min_a + 1, -static_cast<int64_t>(q) * 1000, -1,
        0,     1,         static_cast<int64_t>(q) * 1000,  max_a - 1,
        max_a,
    };

    for (int64_t a : test_a) {
      int32_t lazy = MontArith<32>::MontReduceLazy(a, q, q_inv);
      // Lazy output must be strictly in (-q, q).
      EXPECT_GT(lazy, -static_cast<int32_t>(q));
      EXPECT_LT(lazy, static_cast<int32_t>(q));

      uint32_t canonical = MontArith<32>::MontReduce(a, q, q_inv);
      EXPECT_LT(canonical, q);

      // Verify canonical == (a * R^-1) mod q using APInt.
      APInt a_ap(64, static_cast<uint64_t>(a), /*isSigned=*/true);
      APInt q_ap(64, q);
      APInt r_inv_ap = r_inv.zext(64);
      APInt a_rem = a_ap.srem(q_ap);
      if (a_rem.isNegative()) a_rem += q_ap;
      APInt exp = (a_rem * r_inv_ap).urem(q_ap);

      EXPECT_EQ(canonical, exp.getZExtValue())
          << "MontReduce failed for a=" << a << ", q=" << q;
    }
  }
}

TEST(MontArithTest32, BitReverse) {
  // Test known bit patterns.
  EXPECT_EQ((MontArith<32>::BitReverse(0b0001u, 4)), 0b1000u);
  EXPECT_EQ((MontArith<32>::BitReverse(0b0011u, 4)), 0b1100u);
  EXPECT_EQ((MontArith<32>::BitReverse(0b1010u, 4)), 0b0101u);
  EXPECT_EQ((MontArith<32>::BitReverse(12345u, 0)), 0u);

  // Involutive test for all bit widths 1..32.
  for (unsigned bits = 1; bits <= 32; ++bits) {
    uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    for (uint32_t x : {0u, 1u, mask, mask >> 1, 0x55555555u & mask}) {
      uint32_t rev = MontArith<32>::BitReverse(x, bits);
      EXPECT_EQ(MontArith<32>::BitReverse(rev, bits), x)
          << "BitReverse not involutive for x=" << x << ", bits=" << bits;
    }
  }
}

TEST(MontArithTest32, MultMontBruteForce10To6) {
  // Run 10^6 random pairs distributed across the 32-bit test primes.
  std::mt19937_64 rng(0x123456789ULL);

  const size_t kTotalIterations = 1'000'000;
  const size_t kNumPrimes = sizeof(kPrimes32) / sizeof(kPrimes32[0]);
  const size_t kPerPrime = kTotalIterations / kNumPrimes;

  for (uint32_t q : kPrimes32) {
    int32_t q_inv = MontArith<32>::ComputeQInv(q);
    APInt r_inv_ap = ComputeRInvAPInt(32, APInt(32, q));
    uint32_t r_inv = static_cast<uint32_t>(r_inv_ap.getZExtValue());

    // Edge pairs
    std::vector<std::pair<uint32_t, uint32_t>> edge_pairs = {
        {0, 0},     {0, 1},         {1, 0},     {1, 1},     {q - 1, 1},
        {1, q - 1}, {q - 1, q - 1}, {0, q - 1}, {q - 1, 0}, {q / 2, q / 2}};
    for (const auto& [a, b] : edge_pairs) {
      uint32_t actual = MontArith<32>::MultMont(a, b, q, q_inv);
      uint32_t expected = MultMontOracleAPInt<32>(a, b, q, r_inv_ap);
      ASSERT_EQ(actual, expected)
          << "Edge pair failed for a=" << a << ", b=" << b << ", q=" << q;
    }

    // Random pairs
    std::uniform_int_distribution<uint32_t> dist(0, q - 1);
    for (size_t i = 0; i < kPerPrime; ++i) {
      uint32_t a = dist(rng);
      uint32_t b = dist(rng);
      uint32_t actual = MontArith<32>::MultMont(a, b, q, q_inv);

      // Fast oracle via unsigned __int128:
      // ((a * b) % q * r_inv) % q
      unsigned __int128 prod = static_cast<unsigned __int128>(a) * b;
      uint32_t ab_mod_q = static_cast<uint32_t>(prod % q);
      unsigned __int128 expected_wide =
          static_cast<unsigned __int128>(ab_mod_q) * r_inv;
      uint32_t expected = static_cast<uint32_t>(expected_wide % q);

      ASSERT_EQ(actual, expected)
          << "Random pair failed for a=" << a << ", b=" << b << ", q=" << q;
    }
  }
}

// -----------------------------------------------------------------------------
// 64-bit tests
// -----------------------------------------------------------------------------

TEST(MontArithTest64, ComputeQInv) {
  for (uint64_t q : kPrimes64) {
    int64_t q_inv = MontArith<64>::ComputeQInv(q);
    // Verified: (q * q_inv) == 1 mod 2^64.
    EXPECT_EQ(static_cast<uint64_t>(q_inv) * q, 1ULL)
        << "Failed for prime q = " << q;
  }
}

TEST(MontArithTest64, ComputeRAndR2) {
  for (uint64_t q : kPrimes64) {
    uint64_t r = MontArith<64>::ComputeR(q);
    unsigned __int128 one_128 = 1;
    uint64_t expected_r = static_cast<uint64_t>((one_128 << 64) % q);
    EXPECT_EQ(r, expected_r) << "Failed ComputeR for q = " << q;

    uint64_t r2 = MontArith<64>::ComputeR2(q);
    uint64_t expected_r2 =
        static_cast<uint64_t>((static_cast<unsigned __int128>(r) * r) % q);
    EXPECT_EQ(r2, expected_r2) << "Failed ComputeR2 for q = " << q;
  }
}

TEST(MontArithTest64, AddSubNegateNormalize) {
  for (uint64_t q : kPrimes64) {
    std::vector<uint64_t> test_vals = {0,           1,     2,    q / 2,
                                       (q / 2) + 1, q - 2, q - 1};

    for (uint64_t a : test_vals) {
      // Negate
      uint64_t neg_a = MontArith<64>::Negate(a, q);
      uint64_t exp_neg = (a == 0) ? 0 : q - a;
      EXPECT_EQ(neg_a, exp_neg) << "Negate failed for a=" << a << ", q=" << q;

      // Normalize: output in [-(q-1)/2, (q-1)/2]
      int64_t norm_a = MontArith<64>::Normalize(a, q);
      int64_t exp_norm = (a > (q >> 1))
                             ? static_cast<int64_t>(a) - static_cast<int64_t>(q)
                             : static_cast<int64_t>(a);
      EXPECT_EQ(norm_a, exp_norm)
          << "Normalize failed for a=" << a << ", q=" << q;
      EXPECT_GE(norm_a, -static_cast<int64_t>(q >> 1));
      EXPECT_LE(norm_a, static_cast<int64_t>(q >> 1));

      for (uint64_t b : test_vals) {
        // Add
        uint64_t sum = MontArith<64>::Add(a, b, q);
        uint64_t exp_sum =
            static_cast<uint64_t>((static_cast<unsigned __int128>(a) + b) % q);
        EXPECT_EQ(sum, exp_sum)
            << "Add failed for a=" << a << ", b=" << b << ", q=" << q;

        // Sub
        uint64_t diff = MontArith<64>::Sub(a, b, q);
        uint64_t exp_diff = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(a) + q - (b % q)) % q);
        EXPECT_EQ(diff, exp_diff)
            << "Sub failed for a=" << a << ", b=" << b << ", q=" << q;
      }
    }
  }
}

TEST(MontArithTest64, ToMontFromMontRoundTrip) {
  for (uint64_t q : kPrimes64) {
    int64_t q_inv = MontArith<64>::ComputeQInv(q);
    uint64_t r2 = MontArith<64>::ComputeR2(q);

    std::vector<uint64_t> test_vals = {0, 1, 2, q / 2, q - 1};
    for (uint64_t a : test_vals) {
      uint64_t a_mont = MontArith<64>::ToMont(a, q);
      unsigned __int128 a_128 = a;
      uint64_t expected_mont = static_cast<uint64_t>((a_128 << 64) % q);
      EXPECT_EQ(a_mont, expected_mont)
          << "ToMont failed for a=" << a << ", q=" << q;

      uint64_t a_mont_via_r2 = MontArith<64>::ToMont(a, r2, q, q_inv);
      EXPECT_EQ(a_mont_via_r2, a_mont);

      uint64_t recovered = MontArith<64>::FromMont(a_mont, q, q_inv);
      EXPECT_EQ(recovered, a)
          << "FromMont roundtrip failed for a=" << a << ", q=" << q;
    }
  }
}

TEST(MontArithTest64, MontReduceLazyRangeAndValues) {
  for (uint64_t q : kPrimes64) {
    int64_t q_inv = MontArith<64>::ComputeQInv(q);
    APInt r_inv = ComputeRInvAPInt(64, APInt(64, q));

    __int128 half_base = static_cast<__int128>(1) << 63;
    __int128 min_a = -static_cast<__int128>(q) * half_base;
    __int128 max_a = static_cast<__int128>(q) * half_base - 1;

    std::vector<__int128> test_a = {
        min_a, min_a + 1, -static_cast<__int128>(q) * 1000, -1,
        0,     1,         static_cast<__int128>(q) * 1000,  max_a - 1,
        max_a,
    };

    for (__int128 a : test_a) {
      int64_t lazy = MontArith<64>::MontReduceLazy(a, q, q_inv);
      // Lazy output must be strictly in (-q, q).
      EXPECT_GT(lazy, -static_cast<int64_t>(q));
      EXPECT_LT(lazy, static_cast<int64_t>(q));

      uint64_t canonical = MontArith<64>::MontReduce(a, q, q_inv);
      EXPECT_LT(canonical, q);

      // Verify canonical == (a * R^-1) mod q using APInt.
      // Convert __int128 to APInt (128-bit signed)
      uint64_t a_lo = static_cast<uint64_t>(a);
      uint64_t a_hi = static_cast<uint64_t>(a >> 64);
      uint64_t words[2] = {a_lo, a_hi};
      APInt a_ap(128, llvm::ArrayRef<uint64_t>(words, 2));
      APInt q_ap(128, q);
      APInt r_inv_ap = r_inv.zext(128);
      APInt a_rem = a_ap.srem(q_ap);
      if (a_rem.isNegative()) a_rem += q_ap;
      APInt exp = (a_rem * r_inv_ap).urem(q_ap);

      EXPECT_EQ(canonical, exp.getZExtValue())
          << "MontReduce failed for q=" << q;
    }
  }
}

TEST(MontArithTest64, BitReverse) {
  // Involutive test for all bit widths 1..64.
  for (unsigned bits = 1; bits <= 64; ++bits) {
    uint64_t mask =
        (bits == 64) ? ~uint64_t{0} : ((uint64_t{1} << bits) - uint64_t{1});
    for (uint64_t x : {uint64_t{0}, uint64_t{1}, mask, mask >> 1,
                       static_cast<uint64_t>(0x5555555555555555ULL & mask)}) {
      uint64_t rev = MontArith<64>::BitReverse(x, bits);
      EXPECT_EQ(MontArith<64>::BitReverse(rev, bits), x)
          << "BitReverse not involutive for x=" << x << ", bits=" << bits;
    }
  }
}

TEST(MontArithTest64, MultMontBruteForce10To6) {
  // Run 10^6 random pairs distributed across the 64-bit test primes.
  std::mt19937_64 rng(0x987654321ULL);

  const size_t kTotalIterations = 1'000'000;
  const size_t kNumPrimes = sizeof(kPrimes64) / sizeof(kPrimes64[0]);
  const size_t kPerPrime = kTotalIterations / kNumPrimes;

  for (uint64_t q : kPrimes64) {
    int64_t q_inv = MontArith<64>::ComputeQInv(q);
    APInt r_inv_ap = ComputeRInvAPInt(64, APInt(64, q));
    uint64_t r_inv = r_inv_ap.getZExtValue();

    // Edge pairs
    std::vector<std::pair<uint64_t, uint64_t>> edge_pairs = {
        {0, 0},     {0, 1},         {1, 0},     {1, 1},     {q - 1, 1},
        {1, q - 1}, {q - 1, q - 1}, {0, q - 1}, {q - 1, 0}, {q / 2, q / 2}};
    for (const auto& [a, b] : edge_pairs) {
      uint64_t actual = MontArith<64>::MultMont(a, b, q, q_inv);
      uint64_t expected = MultMontOracleAPInt<64>(a, b, q, r_inv_ap);
      ASSERT_EQ(actual, expected)
          << "Edge pair failed for a=" << a << ", b=" << b << ", q=" << q;
    }

    // Random pairs
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    for (size_t i = 0; i < kPerPrime; ++i) {
      uint64_t a = dist(rng);
      uint64_t b = dist(rng);
      uint64_t actual = MontArith<64>::MultMont(a, b, q, q_inv);

      // Fast oracle via unsigned __int128:
      // ((a * b) % q * r_inv) % q
      unsigned __int128 prod = static_cast<unsigned __int128>(a) * b;
      uint64_t ab_mod_q = static_cast<uint64_t>(prod % q);
      unsigned __int128 expected_wide =
          static_cast<unsigned __int128>(ab_mod_q) * r_inv;
      uint64_t expected = static_cast<uint64_t>(expected_wide % q);

      ASSERT_EQ(actual, expected)
          << "Random pair failed for a=" << a << ", b=" << b << ", q=" << q;
    }
  }
}

}  // namespace
}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir
