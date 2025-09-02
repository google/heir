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

TEST(KernelImplementationTest, HaleviShoup3x5) {
  // Original matrix:
  // [ 0,  1,  2,  3,  4]
  // [ 5,  6,  7,  8,  9]
  // [10, 11, 12, 13, 14]
  //
  // Padded to 4x8:
  // [ 0,  1,  2,  3,  4, 0, 0, 0]
  // [ 5,  6,  7,  8,  9, 0, 0, 0]
  // [10, 11, 12, 13, 14, 0, 0, 0]
  // [ 0,  0,  0,  0,  0, 0, 0, 0]
  //
  // Diagonalized 8x8 matrix:
  std::vector<std::vector<int>> matrix = {{0, 6, 12, 0, 4, 0, 0, 0},
                                          {1, 7, 13, 0, 0, 0, 0, 0},
                                          {2, 8, 14, 0, 0, 0, 10, 0},
                                          {3, 9, 0, 0, 0, 5, 11, 0}};
  // Original vector {0, 1, 2, 3, 4} padded to size 8.
  std::vector<int> vector = {0, 1, 2, 3, 4, 0, 0, 0};
  std::vector<int> expected = {30, 80, 130};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(vectorInput, matrixInput, {3, 5});
  LiteralValue result = evalKernel(dag);
  auto actual = std::get<std::vector<int>>(result.getTensor());

  // The result is of size 8, but we only care about the first 3 elements.
  EXPECT_EQ(std::vector<int>(actual.begin(), actual.begin() + 3), expected);
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
