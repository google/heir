#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/TestingUtils.h"

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

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
                          int64_t period, int64_t n,
                          std::map<int, bool> zeroDiagonals = {}) {
  if (vec.empty()) return vec;
  std::vector<int> result(vec.size(), 0);
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int> rotated = rotate(vec, period * i);
    for (size_t j = 0; j < vec.size(); ++j) {
      if (!zeroDiagonals.contains(i)) {
        result[j] += plaintexts[i][j] * rotated[j];
      }
    }
  }
  return result;
}

std::pair<std::vector<int>, int> runImpl(
    const std::vector<int>& vec,
    const std::vector<std::vector<int>>& plaintexts, int64_t period, int64_t n,
    std::map<int, bool> zeroDiagonals = {},
    const std::string& reduceOp = "arith.addi") {
  if (vec.empty()) return std::make_pair(vec, 0.0);
  LiteralValue vectorInput(vec);

  std::shared_ptr<ArithmeticDagNode<LiteralValue>> result;
  std::optional<LiteralValue> plaintextsInput = std::nullopt;
  if (!plaintexts.empty()) {
    plaintextsInput = std::optional<LiteralValue>(LiteralValue(plaintexts));
  }
  result = implementRotateAndReduce(vectorInput, plaintextsInput, period, n,
                                    zeroDiagonals, reduceOp);
  std::vector<int> resultTensor =
      std::get<std::vector<int>>(evalKernel(result)[0].get());
  int depth = evalMultiplicativeDepth(result);
  return std::make_pair(resultTensor, depth);
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
  std::pair<std::vector<int>, int> actualAndDepth =
      runImpl(vector, plaintexts, period, steps);

  EXPECT_EQ(expected, actualAndDepth.first);
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
  std::pair<std::vector<int>, int> actualAndDepth =
      runImpl(vector, {}, period, steps);
  EXPECT_EQ(expected, actualAndDepth.first);
}

// Property: Implementation should match naive algorithm with plaintexts with
// zero diagonals
void rotateAndReduceWithZeroDiagonalsPlaintexts(
    std::tuple<std::vector<int>, int64_t, int64_t,
               std::vector<std::vector<int>>, std::vector<int>>
        args) {
  const auto& [vector, period, steps, plaintexts, zeroDiagonals] = args;
  std::map<int, bool> zeroDiagonalsMap;
  for (int diagonal : zeroDiagonals) {
    zeroDiagonalsMap[diagonal] = true;
  }
  std::vector<std::vector<int>> zeroedPlaintexts = plaintexts;
  for (int i = 0; i < zeroedPlaintexts.size(); ++i) {
    for (int j = 0; j < zeroedPlaintexts[i].size(); ++j) {
      if (zeroDiagonalsMap.contains(i)) {
        zeroedPlaintexts[i][j] = 0;
      } else {
        zeroedPlaintexts[i][j] = plaintexts[i][j];
      }
    }
  }

  std::vector<int> expected =
      runNaive(vector, zeroedPlaintexts, period, steps, zeroDiagonalsMap);
  std::pair<std::vector<int>, int> actualAndDepthWithNoMap =
      runImpl(vector, zeroedPlaintexts, period, steps);
  std::pair<std::vector<int>, int> actualAndDepth =
      runImpl(vector, zeroedPlaintexts, period, steps, zeroDiagonalsMap);

  EXPECT_EQ(expected, actualAndDepth.first);
  EXPECT_EQ(expected, actualAndDepthWithNoMap.first);

  if (actualAndDepth.second != 0) {
    EXPECT_EQ(actualAndDepth.second, actualAndDepthWithNoMap.second);
  } else {
    // With no filter-list for the zero diagonals, the depth will still be one.
    EXPECT_EQ(actualAndDepthWithNoMap.second, 1);
  }
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
FUZZ_TEST(RotateAndReduceFuzzTest, rotateAndReduceWithoutPlaintexts)
    .WithDomains(
        /*vector=*/fuzztest::VectorOf(fuzztest::InRange(-100, 100))
            .WithMinSize(1)
            .WithMaxSize(32),
        /*period=*/fuzztest::InRange(1L, 4L),
        /*steps=*/fuzztest::ElementOf({1L, 2L, 4L, 8L, 16L}));

// Fuzz test for rotate and reduce with plaintexts and zero diagonals
FUZZ_TEST(RotateAndReduceFuzzTest, rotateAndReduceWithZeroDiagonalsPlaintexts)
    .WithDomains(fuzztest::FlatMap(
        [](size_t vectorSize, int64_t period, int64_t steps) {
          return fuzztest::TupleOf(
              /*vector=*/fuzztest::VectorOf(fuzztest::InRange(-100, 100))
                  .WithSize(vectorSize),
              /*period=*/fuzztest::Just(period),
              /*steps=*/fuzztest::Just(steps),
              /*plaintexts=*/
              fuzztest::VectorOf(
                  fuzztest::VectorOf(fuzztest::InRange(-100, 100))
                      .WithSize(vectorSize))
                  .WithSize(steps),
              /*zeroDiagonals=*/
              fuzztest::VectorOf(fuzztest::InRange<int>(0, steps - 1))
                  .WithMinSize(1)
                  .WithMaxSize(steps));
        },
        /*vectorSize=*/fuzztest::InRange<size_t>(1, 32),
        /*period=*/fuzztest::InRange(1L, 4L),
        /*steps=*/fuzztest::InRange(1L, 16L)));

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
