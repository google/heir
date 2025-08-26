#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Kernel/TestingUtils.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

TEST(KernelImplementationTest, TestHaleviShoupMatvec) {
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<int> expected = {14, 38, 62, 86};
  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag =
      implementMatvec(KernelName::MatvecDiagonal, matrixInput, vectorInput);
  LiteralValue actual = evalKernel(dag);
  EXPECT_EQ(std::get<std::vector<int>>(actual.getTensor()), expected);
}

TEST(KernelImplementationTest, TestExtract) {
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<int> expected = {1, 6, 11, 12};
  LiteralValue matrixInput(matrix);

  auto dag = ArithmeticDagNode<LiteralValue>::extract(
      ArithmeticDagNode<LiteralValue>::leaf(matrixInput), 1);
  LiteralValue actual = evalKernel(dag);
  EXPECT_EQ(std::get<std::vector<int>>(actual.getTensor()), expected);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
