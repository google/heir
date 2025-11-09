#include <iostream>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Kernel/RotationCountVisitor.h"
#include "lib/Kernel/TestingUtils.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project

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

TEST(KernelImplementationTest, TestHaleviShoupMatvecWithLayout) {
  MLIRContext context;
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  std::vector<std::vector<int>> matrix = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  auto diagonalLayout = getDiagonalLayoutRelation(
      RankedTensorType::get({4, 4}, IndexType::get(&context)), 4);
  std::vector<std::vector<int>> diagonalMatrix =
      evaluateLayoutOnMatrix(diagonalLayout, matrix);

  std::vector<int> expected = {14, 38, 62, 86};
  LiteralValue matrixInput = diagonalMatrix;
  LiteralValue vectorInput = vector;

  auto dag =
      implementMatvec(KernelName::MatvecDiagonal, matrixInput, vectorInput);
  LiteralValue actual = evalKernel(dag);
  EXPECT_EQ(std::get<std::vector<int>>(actual.getTensor()), expected);
}

TEST(KernelImplementationTest, Test2DConvWithLayout) {
  MLIRContext context;
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, IndexType::get(&context));

  // 3x3 input data, 2x2 filter
  std::vector<std::vector<int>> data = {{1, -1, 0}, {-3, 0, 2}, {8, 9, 1}};
  std::vector<std::vector<int>> matrix = {{1, -1}, {-1, 1}};

  auto dataLayout = getRowMajorLayoutRelation(dataType, 16);
  std::vector<std::vector<int>> packedData =
      evaluateLayoutOnMatrix(dataLayout, data);

  auto filterLayout = get2dConvFilterRelation(filterType, dataType, 0);
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnMatrix(filterLayout, matrix);

  auto matrixLayout = get2dConvFilterDiagonalizedRelation(filterType, dataType,
                                                          /*padding=*/0, 16)
                          .value();
  std::vector<std::vector<int>> packedMatrix =
      evaluateLayoutOnMatrix(matrixLayout, matrix);
  RankedTensorType expandedMatrixType =
      get2dConvFilterExpandedType(filterType, dataType, /*padding=*/0);

  std::vector<int> expected = {5, 1, -2, -10};
  LiteralValue matrixInput = packedMatrix;
  LiteralValue vectorInput = packedData[0];

  auto dag = implementHaleviShoup(vectorInput, matrixInput,
                                  expandedMatrixType.getShape());
  LiteralValue actual = evalKernel(dag);
  // Result is a 2x2 tensor repeated row-major in a tensor of size 16.
  std::vector<int> actualVector =
      std::get<std::vector<int>>(actual.getTensor());
  std::vector<int> extractedResult = {actualVector.begin(),
                                      actualVector.begin() + 4};
  EXPECT_EQ(extractedResult, expected);
}

TEST(KernelImplementationTest, BicyclicMatmul) {
  MLIRContext context;
  std::vector<std::vector<int>> matrixA = {
      {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  std::vector<std::vector<int>> matrixB = {
      {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
  int m = 3;
  int n = 5;
  int p = 2;
  int numSlots = m * n * p;

  auto layoutA = getBicyclicLayoutRelation(
      RankedTensorType::get({m, n}, IndexType::get(&context)), numSlots);
  auto packedA = evaluateLayoutOnMatrix(layoutA, matrixA);

  auto layoutB = getBicyclicLayoutRelation(
      RankedTensorType::get({n, p}, IndexType::get(&context)), numSlots);
  auto packedB = evaluateLayoutOnMatrix(layoutB, matrixB);

  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  auto dag = implementBicyclicMatmul(packedAValue, packedBValue, m, n, p);
  LiteralValue result = evalKernel(dag);
  auto resultVec = std::get<std::vector<int>>(result.getTensor());

  auto resultLayout = getBicyclicLayoutRelation(
      RankedTensorType::get({m, p}, IndexType::get(&context)), numSlots);
  auto unpackedResult =
      unpackLayoutToMatrix<int>(resultLayout, {resultVec}, {m, p});

  std::vector<std::vector<int>> expected = {{95, 110}, {220, 260}, {345, 410}};
  EXPECT_EQ(unpackedResult, expected);
}

TEST(KernelImplementationTest, BicyclicMatmulRotationCount) {
  MLIRContext context;

  // All triples (x, x+1, x+2) where x is odd are pairwise coprime.
  int m = 123;
  int n = 124;
  int p = 125;

  SymbolicValue packedAValue({m, n}, true);
  SymbolicValue packedBValue({n, p}, true);
  auto dag = implementBicyclicMatmul(packedAValue, packedBValue, m, n, p);

  RotationCountVisitor rotationCounter;
  int64_t rotationCount = rotationCounter.process(dag);

  // 124 + 2*math.sqrt(124) - 3 = 143
  // Actual value is 158.
  //
  // TODO(#2162): Fix the baby/giant step size selector, along with the
  // required modifications to the kernel, so that this bound (and a more
  // extreme bound like a prime m=137) still achieves a roughly n + 2sqrt(n)
  // rotation count.
  EXPECT_EQ(rotationCount, 158);
}

TEST(KernelImplementationTest, TricyclicBatchMatmul) {
  MLIRContext context;

  // Use a small example: h=2, m=3, n=5, p=2 (keeps sizes small for test)
  int64_t h = 2;
  int64_t m = 3;
  int64_t n = 5;
  int64_t p = 2;
  int64_t numSlots = h * m * n;  // single ciphertext for simplicity

  // Build simple batch tensors A (h x m x n) and B (h x n x p) with distinct
  // values
  std::vector<std::vector<std::vector<int>>> A(
      h, std::vector<std::vector<int>>(m, std::vector<int>(n, 0)));
  std::vector<std::vector<std::vector<int>>> B(
      h, std::vector<std::vector<int>>(n, std::vector<int>(p, 0)));

  int val = 1;
  for (int ih = 0; ih < h; ++ih) {
    for (int im = 0; im < m; ++im) {
      for (int in = 0; in < n; ++in) {
        A[ih][im][in] = val++;
      }
    }
  }
  val = 101;
  for (int ih = 0; ih < h; ++ih) {
    for (int in = 0; in < n; ++in) {
      for (int ip = 0; ip < p; ++ip) {
        B[ih][in][ip] = val++;
      }
    }
  }

  // Pack using the tricyclic layout relation.
  RankedTensorType typeA =
      RankedTensorType::get({h, m, n}, IndexType::get(&context));
  RankedTensorType typeB =
      RankedTensorType::get({h, n, p}, IndexType::get(&context));

  auto layoutA = getTricyclicLayoutRelation(typeA, numSlots);
  auto packedA = evaluateLayoutOnTensor(layoutA, A);

  auto layoutB = getTricyclicLayoutRelation(typeB, numSlots);
  auto packedB = evaluateLayoutOnTensor(layoutB, B);

  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  // Generate kernel DAG and evaluate
  auto dag = implementTricyclicBatchCiphertextMatmul(packedAValue, packedBValue,
                                                     h, m, n, p);
  LiteralValue result = evalKernel(dag);
  auto resultVec = std::get<std::vector<int>>(result.getTensor());

  // Unpack result into h x m x p tensor
  auto resultLayout = getTricyclicLayoutRelation(
      RankedTensorType::get({h, m, p}, IndexType::get(&context)), numSlots);
  auto unpackedResult =
      unpackLayoutToTensor<int>(resultLayout, {resultVec}, {h, m, p});

  // Compute expected cleartext batch matmul
  std::vector<std::vector<std::vector<int>>> expected(
      h, std::vector<std::vector<int>>(m, std::vector<int>(p, 0)));
  for (int ih = 0; ih < h; ++ih) {
    for (int im = 0; im < m; ++im) {
      for (int ip = 0; ip < p; ++ip) {
        int sum = 0;
        for (int ic = 0; ic < n; ++ic) {
          sum += A[ih][im][ic] * B[ih][ic][ip];
        }
        expected[ih][im][ip] = sum;
      }
    }
  }

  EXPECT_EQ(unpackedResult, expected);
}
}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
