#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/CheddarRef/MontArith.h"
#include "tests/llvm_runner/memref_types.h"

using mlir::heir::cheddar_ref::MontArith;

constexpr size_t kNumElements = 2048;

extern "C" {
// Prime 1 (32-bit: 974258177)
void _mlir_ciface_test_mont_mul_p1(StridedMemRefType<int32_t, 1>* result,
                                   StridedMemRefType<int32_t, 1>* a,
                                   StridedMemRefType<int32_t, 1>* b);
void _mlir_ciface_test_mont_mul_lazy_p1(StridedMemRefType<int32_t, 1>* result,
                                        StridedMemRefType<int32_t, 1>* a,
                                        StridedMemRefType<int32_t, 1>* b);
void _mlir_ciface_test_mont_reduce_p1(StridedMemRefType<int32_t, 1>* result,
                                      StridedMemRefType<int64_t, 1>* w);
void _mlir_ciface_test_mont_reduce_lazy_p1(
    StridedMemRefType<int32_t, 1>* result, StridedMemRefType<int64_t, 1>* w);
void _mlir_ciface_test_to_mont_p1(StridedMemRefType<int32_t, 1>* result,
                                  StridedMemRefType<int32_t, 1>* a);
void _mlir_ciface_test_from_mont_p1(StridedMemRefType<int32_t, 1>* result,
                                    StridedMemRefType<int32_t, 1>* a);

// Prime 2 (32-bit: 1081212929)
void _mlir_ciface_test_mont_mul_p2(StridedMemRefType<int32_t, 1>* result,
                                   StridedMemRefType<int32_t, 1>* a,
                                   StridedMemRefType<int32_t, 1>* b);
void _mlir_ciface_test_mont_mul_lazy_p2(StridedMemRefType<int32_t, 1>* result,
                                        StridedMemRefType<int32_t, 1>* a,
                                        StridedMemRefType<int32_t, 1>* b);
void _mlir_ciface_test_mont_reduce_p2(StridedMemRefType<int32_t, 1>* result,
                                      StridedMemRefType<int64_t, 1>* w);
void _mlir_ciface_test_mont_reduce_lazy_p2(
    StridedMemRefType<int32_t, 1>* result, StridedMemRefType<int64_t, 1>* w);
void _mlir_ciface_test_to_mont_p2(StridedMemRefType<int32_t, 1>* result,
                                  StridedMemRefType<int32_t, 1>* a);
void _mlir_ciface_test_from_mont_p2(StridedMemRefType<int32_t, 1>* result,
                                    StridedMemRefType<int32_t, 1>* a);

// Prime 3 (64-bit: 1125899908022273)
void _mlir_ciface_test_mont_mul_p3(StridedMemRefType<int64_t, 1>* result,
                                   StridedMemRefType<int64_t, 1>* a,
                                   StridedMemRefType<int64_t, 1>* b);
void _mlir_ciface_test_mont_mul_lazy_p3(StridedMemRefType<int64_t, 1>* result,
                                        StridedMemRefType<int64_t, 1>* a,
                                        StridedMemRefType<int64_t, 1>* b);
void _mlir_ciface_test_mont_reduce_p3(StridedMemRefType<int64_t, 1>* result,
                                      StridedMemRefType<__int128_t, 1>* w);
void _mlir_ciface_test_mont_reduce_lazy_p3(
    StridedMemRefType<int64_t, 1>* result, StridedMemRefType<__int128_t, 1>* w);
void _mlir_ciface_test_to_mont_p3(StridedMemRefType<int64_t, 1>* result,
                                  StridedMemRefType<int64_t, 1>* a);
void _mlir_ciface_test_from_mont_p3(StridedMemRefType<int64_t, 1>* result,
                                    StridedMemRefType<int64_t, 1>* a);
}

template <typename T>
StridedMemRefType<T, 1> MakeMemRef(std::vector<T>& vec) {
  return StridedMemRefType<T, 1>(vec.data(), vec.data(), 0, vec.size(), 1);
}

void Test32BitPrime(uint32_t q,
                    void (*mont_mul)(StridedMemRefType<int32_t, 1>*,
                                     StridedMemRefType<int32_t, 1>*,
                                     StridedMemRefType<int32_t, 1>*),
                    void (*mont_mul_lazy)(StridedMemRefType<int32_t, 1>*,
                                          StridedMemRefType<int32_t, 1>*,
                                          StridedMemRefType<int32_t, 1>*),
                    void (*mont_reduce)(StridedMemRefType<int32_t, 1>*,
                                        StridedMemRefType<int64_t, 1>*),
                    void (*mont_reduce_lazy)(StridedMemRefType<int32_t, 1>*,
                                             StridedMemRefType<int64_t, 1>*),
                    void (*to_mont)(StridedMemRefType<int32_t, 1>*,
                                    StridedMemRefType<int32_t, 1>*),
                    void (*from_mont)(StridedMemRefType<int32_t, 1>*,
                                      StridedMemRefType<int32_t, 1>*)) {
  int32_t q_inv = MontArith<32>::ComputeQInv(q);
  uint32_t r2 = MontArith<32>::ComputeR2(q);

  std::mt19937_64 rng(0x12345678 + q);
  std::uniform_int_distribution<uint32_t> dist_q(0, q - 1);

  std::vector<int32_t> a(kNumElements);
  std::vector<int32_t> b(kNumElements);
  std::vector<int64_t> w(kNumElements);

  int64_t half_base = static_cast<int64_t>(1) << 31;
  int64_t min_w = -static_cast<int64_t>(q) * half_base;
  int64_t max_w = static_cast<int64_t>(q) * half_base - 1;
  std::uniform_int_distribution<int64_t> dist_w(min_w / 2, max_w / 2);

  for (size_t i = 0; i < kNumElements; ++i) {
    a[i] = static_cast<int32_t>(dist_q(rng));
    b[i] = static_cast<int32_t>(dist_q(rng));
    if (i % 2 == 0) {
      w[i] = static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
    } else {
      w[i] = dist_w(rng);
    }
  }

  // 1. mont_mul
  {
    auto in_a = a;
    auto in_b = b;
    auto memref_a = MakeMemRef(in_a);
    auto memref_b = MakeMemRef(in_b);
    StridedMemRefType<int32_t, 1> res;
    mont_mul(&res, &memref_a, &memref_b);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint32_t expected = MontArith<32>::MultMont(
          static_cast<uint32_t>(a[i]), static_cast<uint32_t>(b[i]), q, q_inv);
      EXPECT_EQ(static_cast<uint32_t>(res.data[i]), expected)
          << "mont_mul mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 2. mont_mul lazy
  {
    auto in_a = a;
    auto in_b = b;
    auto memref_a = MakeMemRef(in_a);
    auto memref_b = MakeMemRef(in_b);
    StridedMemRefType<int32_t, 1> res;
    mont_mul_lazy(&res, &memref_a, &memref_b);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      int32_t expected = MontArith<32>::MultMontLazy(a[i], b[i], q, q_inv);
      EXPECT_EQ(res.data[i], expected)
          << "mont_mul_lazy mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 3. mont_reduce
  {
    auto in_w = w;
    auto memref_w = MakeMemRef(in_w);
    StridedMemRefType<int32_t, 1> res;
    mont_reduce(&res, &memref_w);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint32_t expected = MontArith<32>::MontReduce(w[i], q, q_inv);
      EXPECT_EQ(static_cast<uint32_t>(res.data[i]), expected)
          << "mont_reduce mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 4. mont_reduce lazy
  {
    auto in_w = w;
    auto memref_w = MakeMemRef(in_w);
    StridedMemRefType<int32_t, 1> res;
    mont_reduce_lazy(&res, &memref_w);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      int32_t expected = MontArith<32>::MontReduceLazy(w[i], q, q_inv);
      EXPECT_EQ(res.data[i], expected)
          << "mont_reduce_lazy mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 5. to_mont
  {
    auto in_a = a;
    auto memref_a = MakeMemRef(in_a);
    StridedMemRefType<int32_t, 1> res;
    to_mont(&res, &memref_a);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint32_t expected =
          MontArith<32>::ToMont(static_cast<uint32_t>(a[i]), r2, q, q_inv);
      EXPECT_EQ(static_cast<uint32_t>(res.data[i]), expected)
          << "to_mont mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 6. from_mont
  {
    auto in_a = a;
    auto memref_a = MakeMemRef(in_a);
    StridedMemRefType<int32_t, 1> res;
    from_mont(&res, &memref_a);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint32_t expected =
          MontArith<32>::FromMont(static_cast<uint32_t>(a[i]), q, q_inv);
      EXPECT_EQ(static_cast<uint32_t>(res.data[i]), expected)
          << "from_mont mismatch at index " << i;
    }
    std::free(res.basePtr);
  }
}

TEST(MontRunnerTest, Prime1_32bit) {
  Test32BitPrime(974258177, _mlir_ciface_test_mont_mul_p1,
                 _mlir_ciface_test_mont_mul_lazy_p1,
                 _mlir_ciface_test_mont_reduce_p1,
                 _mlir_ciface_test_mont_reduce_lazy_p1,
                 _mlir_ciface_test_to_mont_p1, _mlir_ciface_test_from_mont_p1);
}

TEST(MontRunnerTest, Prime2_32bit) {
  Test32BitPrime(1081212929, _mlir_ciface_test_mont_mul_p2,
                 _mlir_ciface_test_mont_mul_lazy_p2,
                 _mlir_ciface_test_mont_reduce_p2,
                 _mlir_ciface_test_mont_reduce_lazy_p2,
                 _mlir_ciface_test_to_mont_p2, _mlir_ciface_test_from_mont_p2);
}

TEST(MontRunnerTest, Prime3_64bit) {
  uint64_t q = 1125899908022273ULL;
  int64_t q_inv = MontArith<64>::ComputeQInv(q);
  uint64_t r2 = MontArith<64>::ComputeR2(q);

  std::mt19937_64 rng(0x87654321);
  std::uniform_int_distribution<uint64_t> dist_q(0, q - 1);

  std::vector<int64_t> a(kNumElements);
  std::vector<int64_t> b(kNumElements);
  std::vector<__int128_t> w(kNumElements);

  for (size_t i = 0; i < kNumElements; ++i) {
    a[i] = static_cast<int64_t>(dist_q(rng));
    b[i] = static_cast<int64_t>(dist_q(rng));
    if (i % 2 == 0) {
      w[i] = static_cast<__int128_t>(a[i]) * static_cast<__int128_t>(b[i]);
    } else {
      int64_t rnd_hi =
          static_cast<int64_t>(rng()) % (static_cast<int64_t>(q) / 2);
      uint64_t rnd_lo = rng();
      w[i] =
          static_cast<__int128_t>(rnd_hi) * (static_cast<__int128_t>(1) << 64) +
          rnd_lo;
    }
  }

  // 1. mont_mul
  {
    auto in_a = a;
    auto in_b = b;
    auto memref_a = MakeMemRef(in_a);
    auto memref_b = MakeMemRef(in_b);
    StridedMemRefType<int64_t, 1> res;
    _mlir_ciface_test_mont_mul_p3(&res, &memref_a, &memref_b);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint64_t expected = MontArith<64>::MultMont(
          static_cast<uint64_t>(a[i]), static_cast<uint64_t>(b[i]), q, q_inv);
      EXPECT_EQ(static_cast<uint64_t>(res.data[i]), expected)
          << "mont_mul_p3 mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 2. mont_mul lazy
  {
    auto in_a = a;
    auto in_b = b;
    auto memref_a = MakeMemRef(in_a);
    auto memref_b = MakeMemRef(in_b);
    StridedMemRefType<int64_t, 1> res;
    _mlir_ciface_test_mont_mul_lazy_p3(&res, &memref_a, &memref_b);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      int64_t expected = MontArith<64>::MultMontLazy(a[i], b[i], q, q_inv);
      EXPECT_EQ(res.data[i], expected)
          << "mont_mul_lazy_p3 mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 3. mont_reduce
  {
    auto in_w = w;
    auto memref_w = MakeMemRef(in_w);
    StridedMemRefType<int64_t, 1> res;
    _mlir_ciface_test_mont_reduce_p3(&res, &memref_w);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint64_t expected = MontArith<64>::MontReduce(w[i], q, q_inv);
      EXPECT_EQ(static_cast<uint64_t>(res.data[i]), expected)
          << "mont_reduce_p3 mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 4. mont_reduce lazy
  {
    auto in_w = w;
    auto memref_w = MakeMemRef(in_w);
    StridedMemRefType<int64_t, 1> res;
    _mlir_ciface_test_mont_reduce_lazy_p3(&res, &memref_w);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      int64_t expected = MontArith<64>::MontReduceLazy(w[i], q, q_inv);
      EXPECT_EQ(res.data[i], expected)
          << "mont_reduce_lazy_p3 mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 5. to_mont
  {
    auto in_a = a;
    auto memref_a = MakeMemRef(in_a);
    StridedMemRefType<int64_t, 1> res;
    _mlir_ciface_test_to_mont_p3(&res, &memref_a);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint64_t expected =
          MontArith<64>::ToMont(static_cast<uint64_t>(a[i]), r2, q, q_inv);
      EXPECT_EQ(static_cast<uint64_t>(res.data[i]), expected)
          << "to_mont_p3 mismatch at index " << i;
    }
    std::free(res.basePtr);
  }

  // 6. from_mont
  {
    auto in_a = a;
    auto memref_a = MakeMemRef(in_a);
    StridedMemRefType<int64_t, 1> res;
    _mlir_ciface_test_from_mont_p3(&res, &memref_a);
    ASSERT_EQ(res.sizes[0], kNumElements);
    for (size_t i = 0; i < kNumElements; ++i) {
      uint64_t expected =
          MontArith<64>::FromMont(static_cast<uint64_t>(a[i]), q, q_inv);
      EXPECT_EQ(static_cast<uint64_t>(res.data[i]), expected)
          << "from_mont_p3 mismatch at index " << i;
    }
    std::free(res.basePtr);
  }
}
