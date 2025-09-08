#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "fuzztest/fuzztest.h"  // from @fuzztest
#include "gtest/gtest.h"        // from @googletest
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/TestingUtils.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

// Helper function to rotate a vector left by a given amount
std::vector<int> rotate(const std::vector<int>& vec, int64_t amount) {
  if (vec.empty()) return vec;
  int64_t n = vec.size();
  std::vector<int> result(n);
  for (int64_t i = 0; i < n; ++i) {
    result[i] = vec[(i + amount) % n];
  }
  return result;
}

std::vector<int> runNaive(const std::vector<int>& vec,
                          const std::vector<std::vector<int>>& plaintexts,
                          int64_t period, int64_t n) {
  if (vec.empty()) return vec;
  std::vector<int> result(vec.size(), 0);
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int> rotated = rotate(vec, period * i);
    for (size_t j = 0; j < vec.size(); ++j) {
      result[j] += plaintexts[i][j] * rotated[j];
    }
  }
  return result;
}

std::vector<int> runImpl(const std::vector<int>& vec,
                         const std::vector<std::vector<int>>& plaintexts,
                         int64_t period, int64_t n,
                         const std::string& reduceOp = "arith.addi") {
  if (vec.empty()) return vec;
  LiteralValue vectorInput(vec);

  std::shared_ptr<ArithmeticDagNode<LiteralValue>> result;
  std::optional<LiteralValue> plaintextsInput = std::nullopt;
  if (!plaintexts.empty()) {
    plaintextsInput = std::optional<LiteralValue>(LiteralValue(plaintexts));
  }
  result = implementRotateAndReduce(vectorInput, plaintextsInput, period, n,
                                    reduceOp);
  return std::get<std::vector<int>>(evalKernel(result).getTensor());
}

// Property: Implementation should match naive algorithm with plaintexts
void rotateAndReduceWithPlaintextsMatchesNaive(
    const std::vector<int>& vector, int64_t period, int64_t steps,
    const std::vector<std::vector<int>>& plaintexts) {
  if (vector.empty() || steps <= 0 || period <= 0) return;
  if (plaintexts.size() != static_cast<size_t>(steps)) return;
  for (const auto& plaintext : plaintexts) {
    if (plaintext.size() != vector.size()) return;
  }

  std::vector<int> expected = runNaive(vector, plaintexts, period, steps);
  std::vector<int> actual = runImpl(vector, plaintexts, period, steps);

  EXPECT_EQ(expected, actual);
}

// Naive implementation for the no-plaintexts case
std::vector<int> runNaiveNoPlaintexts(const std::vector<int>& vec,
                                      int64_t period, int64_t steps) {
  if (vec.empty()) return vec;
  std::vector<int> result = vec;
  for (int64_t i = 1; i < steps; ++i) {
    std::vector<int> rotated = rotate(vec, period * i);
    for (size_t j = 0; j < vec.size(); ++j) {
      result[j] += rotated[j];
    }
  }
  return result;
}

// Property: Implementation without plaintexts should sum rotations
void rotateAndReduceWithoutPlaintexts(const std::vector<int>& vector,
                                      int64_t period, int64_t steps) {
  if (vector.empty() || steps <= 0 || period <= 0) return;

  std::vector<int> expected = runNaiveNoPlaintexts(vector, period, steps);
  std::vector<int> actual = runImpl(vector, {}, period, steps);
  EXPECT_EQ(expected, actual);
}

// Fuzz test for rotate and reduce with plaintexts
FUZZ_TEST(RotateAndReduceFuzzTest, rotateAndReduceWithPlaintextsMatchesNaive)
    .WithDomains(
        /*vector=*/fuzztest::VectorOf(fuzztest::InRange(-100, 100))
            .WithMinSize(1)
            .WithMaxSize(32),
        /*period=*/fuzztest::InRange(1L, 4L),
        /*steps=*/fuzztest::InRange(1L, 16L),
        /*plaintexts=*/
        fuzztest::VectorOf(fuzztest::VectorOf(fuzztest::InRange(-100, 100))
                               .WithMinSize(1)
                               .WithMaxSize(32))
            .WithMinSize(1)
            .WithMaxSize(16));

// Fuzz test for rotate and reduce without plaintexts
// NOTE: This test currently includes non-power-of-2 values for steps.
// The implementation currently only supports power-of-2 steps, so some
// tests may fail until the implementation is updated to handle arbitrary steps.
FUZZ_TEST(RotateAndReduceFuzzTest, rotateAndReduceWithoutPlaintexts)
    .WithDomains(
        /*vector=*/fuzztest::VectorOf(fuzztest::InRange(-100, 100))
            .WithMinSize(1)
            .WithMaxSize(32),
        /*period=*/fuzztest::InRange(1L, 4L),
        /*steps=*/fuzztest::ElementOf({1L, 2L, 4L, 8L, 16L}));

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
