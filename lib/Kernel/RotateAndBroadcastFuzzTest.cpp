#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/TestingUtils.h"

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace kernel {
namespace {

std::vector<int> runNaive(const std::vector<int>& vec, int64_t dimension,
                          int64_t steps, std::vector<int64_t> originalShape) {
  std::vector<int> result(vec.size(), 0);

  int64_t currentSize = 1;
  for (int64_t i = dimension; i < static_cast<int64_t>(originalShape.size());
       ++i) {
    currentSize *= originalShape[i];
  }
  // Copy the first `currentSize` entries to every position in the dimension.
  int64_t blockSize = currentSize * steps;
  for (int64_t blockStart = 0; blockStart < static_cast<int64_t>(vec.size());
       blockStart += blockSize) {
    for (int64_t t = 0;
         t < blockSize && blockStart + t < static_cast<int64_t>(vec.size());
         ++t) {
      int64_t src = blockStart + (t % currentSize);
      result[blockStart + t] = vec[src];
    }
  }

  return result;
}

std::vector<int> runImpl(const std::vector<int>& vec, int64_t dimension,
                         int64_t steps, ArrayRef<int64_t> originalShape) {
  LiteralValue vectorInput(vec);

  std::shared_ptr<ArithmeticDagNode<LiteralValue>> result;

  result = implementRotateAndBroadcast(
      vectorInput, dimension, steps,
      DagType::intTensor(32, {static_cast<int64_t>(vec.size())}), vec.size(),
      originalShape);

  return std::get<std::vector<int>>(evalKernel(result)[0].get());
}

void rotateAndBroadcastMatchesNaive(const std::vector<int>& vector,
                                    int64_t dimension, int64_t steps,
                                    std::vector<int64_t> originalShape) {
  if (vector.empty() || steps <= 0 || dimension < 0 ||
      dimension >= static_cast<int64_t>(originalShape.size())) {
    return;
  }
  std::vector<int> expected = runNaive(vector, dimension, steps, originalShape);
  std::vector<int> actual = runImpl(vector, dimension, steps, originalShape);
  EXPECT_EQ(expected, actual);
}

auto ValidBroadcastDomains() {
  const int64_t maxDimension = 6;
  const int64_t maxVecSize = 131072;  // 2^17, realistic ciphertext size

  auto numDimsDomain = fuzztest::InRange(int64_t{2}, maxDimension);

  // Generate initial case
  auto shapeParamsDomain = fuzztest::FlatMap(
      [=](int64_t numDims) {
        return fuzztest::TupleOf(
            fuzztest::InRange(int64_t{0}, numDims - 1),
            fuzztest::InRange(int64_t{1}, maxVecSize),
            fuzztest::VectorOf(fuzztest::InRange(int64_t{1}, maxVecSize))
                .WithSize(numDims - 1));
      },
      numDimsDomain);

  // Filter out combinations that are invalid
  auto validShapeParamsDomain = fuzztest::Filter(
      [=](const std::tuple<int64_t, int64_t, std::vector<int64_t>>& params) {
        auto [broadcast, steps, shape] = params;
        int64_t totalSize = steps;
        for (int64_t dim : shape) {
          if (dim <= 0 || totalSize > maxVecSize / dim) return false;
          totalSize *= dim;
        }
        return totalSize <= maxVecSize;
      },
      shapeParamsDomain);

  // Returns a tuple of the parameters
  return fuzztest::FlatMap(
      [](const std::tuple<int64_t, int64_t, std::vector<int64_t>>& params) {
        auto [broadcastDimension, steps, originalShape] = params;

        int64_t required_size = 1;
        for (int64_t dim : originalShape) required_size *= dim;
        required_size *= steps;

        return fuzztest::TupleOf(fuzztest::VectorOf(fuzztest::Arbitrary<int>())
                                     .WithSize(required_size),
                                 fuzztest::Just(broadcastDimension),
                                 fuzztest::Just(steps),
                                 fuzztest::Just(originalShape));
      },
      validShapeParamsDomain);
}

// Proxy to adapt the tuple to 4 parameters with filters
void rotateAndBroadcastProxy(
    const std::tuple<std::vector<int>, int64_t, int64_t, std::vector<int64_t>>&
        all_params) {
  auto [vec, broadcastDimension, steps, originalShape] = all_params;
  rotateAndBroadcastMatchesNaive(vec, broadcastDimension, steps, originalShape);
}

FUZZ_TEST(RotateAndBroadcastFuzzTest, rotateAndBroadcastProxy)
    .WithDomains(ValidBroadcastDomains());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
