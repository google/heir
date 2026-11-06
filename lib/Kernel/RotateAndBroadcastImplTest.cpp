#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/KernelImplementation.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

// A `linalg.broadcast` op computes a broadcast of a tensor such that the output
// at index i in the brodcasted dimension is the input at the first value of
// that dimension. The naive approach would compute the broadcast by copying the
// input value across the dimension using just a copy operation. This is a
// ground truth for testing.

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

TEST(RotateAndReduceImplTest, TestScalarBroadcast) {
  std::vector<int64_t> originalShape = {1};
  int64_t dimension = 0;
  int64_t steps = 8;
  std::vector<int> vector = {2, 100, 100, 100, 100, 100, 100, 100};

  std::vector<int> expected(8, 2);
  std::vector<int> naive = runNaive(vector, dimension, steps, originalShape);
  std::vector<int> impl = runImpl(vector, dimension, steps, originalShape);
  EXPECT_EQ(naive, impl);
  EXPECT_EQ(naive, expected);
}

TEST(RotateAndReduceImplTest, TestNonPowerOfTwoSteps) {
  std::vector<int64_t> originalShape = {2, 3};
  int64_t dimension = 1;
  int64_t steps = 3;

  std::vector<int> vector = {0, 1, 2, 100, 100, 100, 100, 100, 100,
                             3, 4, 5, 100, 100, 100, 100, 100, 100};

  std::vector<int> expected = {0, 1, 2, 0, 1, 2, 0, 1, 2,
                               3, 4, 5, 3, 4, 5, 3, 4, 5};
  std::vector<int> naive = runNaive(vector, dimension, steps, originalShape);
  std::vector<int> impl = runImpl(vector, dimension, steps, originalShape);
  EXPECT_EQ(naive, impl);
  EXPECT_EQ(naive, expected);
}

TEST(RotateAndBroadcastImplTest, TestBroadcastLarger) {
  std::vector<int64_t> originalShape = {2, 2};
  int64_t dimension = 2;
  int64_t steps = 3;
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  std::vector<int> expected = {1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10};
  std::vector<int> naive = runNaive(vector, dimension, steps, originalShape);
  std::vector<int> impl = runImpl(vector, dimension, steps, originalShape);
  EXPECT_EQ(naive, impl);
  EXPECT_EQ(naive, expected);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
