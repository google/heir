#include <iostream>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/TestingUtils.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

// A `tensor_ext.rotate_and_reduce` op computes the reduction of rotated tensors
// in the form `\sum_{i = 0}^{n} P(i) \cdot rotate(v, T*i)` where `T` is some
// period of rotation. The naive approach would compute `n` rotations of the
// ciphertext `v`.

std::vector<int> rotate(const std::vector<int>& vec, int64_t amount) {
  int64_t n = vec.size();
  std::vector<int> result(n);
  for (int64_t i = 0; i < n; ++i) {
    // rotate left cyclically
    result[i] = vec[(i + amount) % n];
  }
  return result;
}

TEST(RotateAndReduceImplTest, TestRotateHelper) {
  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> expected = {2, 3, 4, 5, 6, 7, 0, 1};
  std::vector<int> actual = rotate(vector, 2);
  EXPECT_EQ(expected, actual);
}

std::vector<int> runNaive(const std::vector<int>& vec,
                          const std::vector<std::vector<int>>& plaintexts,
                          int64_t period, int64_t n) {
  std::vector<int> result(vec.size(), 0);
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int> rotated = rotate(vec, period * i);
    for (int64_t j = 0; j < vec.size(); ++j) {
      result[j] += plaintexts[i][j] * rotated[j];
    }
  }
  return result;
}

std::vector<int> runImpl(const std::vector<int>& vec,
                         const std::vector<std::vector<int>>& plaintexts,
                         int64_t period, int64_t n) {
  LiteralValue vectorInput(vec);

  std::shared_ptr<ArithmeticDagNode<LiteralValue>> result;
  std::optional<LiteralValue> plaintextsInput = std::nullopt;
  if (!plaintexts.empty()) {
    plaintextsInput = std::optional<LiteralValue>(LiteralValue(plaintexts));
  }
  result = implementRotateAndReduce(vectorInput, plaintextsInput, period, n);
  std::cerr << "Rotate and reduce dag: " << printKernel(result) << "\n";
  return std::get<std::vector<int>>(evalKernel(result).getTensor());
}

TEST(RotateAndReduceImplTest, TestUnitPeriodWithPlaintextsSimpleValues) {
  int64_t n = 8;
  int64_t period = 1;

  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::vector<int>> plaintexts = {
      {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
  };

  std::vector<int> expected = runNaive(vector, plaintexts, period, n);
  std::vector<int> actual = runImpl(vector, plaintexts, period, n);
  EXPECT_EQ(expected, actual);
}

TEST(RotateAndReduceImplTest, TestUnitPeriodWithPlaintexts) {
  int64_t n = 8;
  int64_t period = 1;

  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::vector<int>> plaintexts = {
      {1, 2, 3, 4, 5, 6, 7, 8},      {2, 3, 4, 5, 6, 7, 8, 9},
      {3, 4, 5, 6, 7, 8, 9, 10},     {4, 5, 6, 7, 8, 9, 10, 11},
      {5, 6, 7, 8, 9, 10, 11, 12},   {6, 7, 8, 9, 10, 11, 12, 13},
      {7, 8, 9, 10, 11, 12, 13, 14}, {8, 9, 10, 11, 12, 13, 14, 15},
  };

  std::vector<int> expected = runNaive(vector, plaintexts, period, n);
  std::vector<int> actual = runImpl(vector, plaintexts, period, n);
  EXPECT_EQ(expected, actual);
}

TEST(RotateAndReduceImplTest, TestPeriod2WithPlaintexts) {
  int64_t n = 8;
  int64_t period = 2;

  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::vector<int>> plaintexts = {
      {1, 2, 3, 4, 5, 6, 7, 8},      {2, 3, 4, 5, 6, 7, 8, 9},
      {3, 4, 5, 6, 7, 8, 9, 10},     {4, 5, 6, 7, 8, 9, 10, 11},
      {5, 6, 7, 8, 9, 10, 11, 12},   {6, 7, 8, 9, 10, 11, 12, 13},
      {7, 8, 9, 10, 11, 12, 13, 14}, {8, 9, 10, 11, 12, 13, 14, 15},
  };

  std::vector<int> expected = runNaive(vector, plaintexts, period, n);
  std::vector<int> actual = runImpl(vector, plaintexts, period, n);
  EXPECT_EQ(expected, actual);
}

TEST(RotateAndReduceImplTest, TestPeriod2WithPlaintextsSmallerN) {
  int64_t n = 4;
  int64_t period = 2;

  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::vector<int>> plaintexts = {
      {1, 2, 3, 4, 5, 6, 7, 8},
      {2, 3, 4, 5, 6, 7, 8, 9},
      {3, 4, 5, 6, 7, 8, 9, 10},
      {4, 5, 6, 7, 8, 9, 10, 11},
  };

  std::vector<int> expected = runNaive(vector, plaintexts, period, n);
  std::vector<int> actual = runImpl(vector, plaintexts, period, n);
  EXPECT_EQ(expected, actual);
}

TEST(RotateAndReduceImplTest, TestPeriod1WithNoPlaintext) {
  int64_t n = 8;
  int64_t period = 1;
  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> expected(8, 28);
  std::vector<int> actual = runImpl(vector, {}, period, n);
  EXPECT_EQ(expected, actual);
}

TEST(RotateAndReduceImplTest, TestPeriod2WithNoPlaintext) {
  int64_t n = 4;
  int64_t period = 2;
  std::vector<int> vector = {0, 1, 2, 3, 4, 5, 6, 7};
  // This test computes the partial sum of entries with a stride of 2, i.e.,
  // every even index contains the sum of all even indices and every odd index
  // contains the sum of all odd indices.
  std::vector<int> expected = {12, 16, 12, 16, 12, 16, 12, 16};
  std::vector<int> actual = runImpl(vector, {}, period, n);
  EXPECT_EQ(expected, actual);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
