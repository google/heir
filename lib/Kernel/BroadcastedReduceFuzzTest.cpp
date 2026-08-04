#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/KernelImplementation.h"

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace kernel {
namespace {

std::vector<int> runNaiveBroadcastedReduce(const std::vector<int>& vec,
                                           int64_t period, int64_t steps) {
  int64_t n = vec.size();
  int64_t B = steps;
  int64_t blockSize = B * period;
  std::vector<int> result(n, 0);

  int64_t numBlocks = n / blockSize;

  for (int64_t k = 0; k < numBlocks; ++k) {
    for (int64_t offset = 0; offset < period; ++offset) {
      int sum = 0;
      for (int64_t i = 0; i < B; ++i) {
        sum += vec[k * blockSize + i * period + offset];
      }
      for (int64_t i = 0; i < B; ++i) {
        result[k * blockSize + i * period + offset] = sum;
      }
    }
  }
  return result;
}

std::vector<int> generateCleanupMask(int64_t numSlots, int64_t period,
                                     int64_t steps) {
  std::vector<int> mask(numSlots, 0);
  int64_t B = steps;
  int64_t blockSize = B * period;
  int64_t numBlocks = numSlots / blockSize;
  for (int64_t k = 0; k < numBlocks; ++k) {
    for (int64_t offset = 0; offset < period; ++offset) {
      mask[k * blockSize + (B - 1) * period + offset] = 1;
    }
  }
  return mask;
}

void broadcastedReduceMatchesNaive(int logN, int logB, int logPeriod,
                                   const std::vector<int>& inputTemplate,
                                   bool unroll) {
  int64_t numSlots = 1 << logN;
  int64_t steps = 1 << logB;
  int64_t period = 1 << logPeriod;

  if (steps * period > numSlots) return;

  // Resize inputTemplate to numSlots
  std::vector<int> vec(numSlots);
  for (int64_t i = 0; i < numSlots; ++i) {
    vec[i] = inputTemplate[i % inputTemplate.size()];
  }

  std::vector<int> expected = runNaiveBroadcastedReduce(vec, period, steps);

  using NodeTy = ArithmeticDagNode<LiteralValue>;
  using NodePtr = std::shared_ptr<NodeTy>;

  LiteralValue vectorInput(vec);
  auto vectorDag = NodeTy::leaf(vectorInput);

  std::optional<NodePtr> cleanupMaskDag = std::nullopt;
  if (steps * period < numSlots) {
    auto mask = generateCleanupMask(numSlots, period, steps);
    cleanupMaskDag = NodeTy::leaf(LiteralValue(mask));
  }

  auto result = implementBroadcastedReduce<LiteralValue>(
      vectorDag, cleanupMaskDag, period, steps, numSlots,
      DagType::intTensor(32, {numSlots}), "arith.addi", unroll);

  std::vector<int> actual =
      std::get<std::vector<int>>(evalKernel(result)[0].get());

  EXPECT_EQ(expected, actual);
}

auto ValidParameters() {
  return fuzztest::FlatMap(
      [](int logN) {
        return fuzztest::FlatMap(
            [logN](int logB) {
              return fuzztest::TupleOf(fuzztest::Just(logN),
                                       fuzztest::Just(logB),
                                       fuzztest::InRange(0, logN - logB));
            },
            fuzztest::InRange(1, logN));
      },
      fuzztest::InRange(3, 7)  // N from 8 to 128
  );
}

void BroadcastedReduceFuzz(const std::tuple<int, int, int>& params,
                           const std::vector<int>& inputTemplate, bool unroll) {
  auto [logN, logB, logPeriod] = params;
  broadcastedReduceMatchesNaive(logN, logB, logPeriod, inputTemplate, unroll);
}

FUZZ_TEST(BroadcastedReduceFuzzTest, BroadcastedReduceFuzz)
    .WithDomains(ValidParameters(),
                 fuzztest::VectorOf(fuzztest::InRange(-100, 100))
                     .WithMinSize(1)
                     .WithMaxSize(128),
                 fuzztest::Arbitrary<bool>());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
