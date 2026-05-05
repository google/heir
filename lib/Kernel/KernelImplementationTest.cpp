#include <cstdint>
#include <functional>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Kernel/RotationCountVisitor.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {
namespace {

using tensor4d = std::vector<std::vector<std::vector<std::vector<int>>>>;

std::function<int(const std::vector<int64_t>&)> getDataValueFn(tensor4d data) {
  return [data](const std::vector<int64_t>& domainPoint) -> int {
    return data[domainPoint[0]][domainPoint[1]][domainPoint[2]][domainPoint[3]];
  };
}

// Parametrize over whether the kernel is unrolled and whether rows are
// interchanged
class KernelImplementationTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(KernelImplementationTest, TestHaleviShoupMatvec) {
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<int> expected = {14, 38, 62, 86};
  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {4, 4}, DagType::intTensor(32, {4}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST_P(KernelImplementationTest, HaleviShoup3x5) {
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

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {3, 5}, DagType::intTensor(32, {8}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  // The result is of size 8, but we only care about the first 3 elements.
  EXPECT_EQ(std::vector<int>(actual.begin(), actual.begin() + 3), expected);
}

TEST_P(KernelImplementationTest, HaleviShoup4x2) {
  // Original matrix (4x2):
  // [0, 1]
  // [2, 3]
  // [4, 5]
  // [6, 7]
  //
  // Pre-packed diagonally:
  // diag_0 = (0, 3, 4, 7)
  // diag_1 = (1, 2, 5, 6)
  std::vector<std::vector<int>> matrix = {{0, 3, 4, 7}, {1, 2, 5, 6}};
  // Replicated vector {3, 4, 3, 4}
  std::vector<int> vector = {3, 4, 3, 4};
  std::vector<int> expected = {4, 18, 32, 46};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {4, 2}, DagType::intTensor(32, {4}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  EXPECT_EQ(actual, expected);
}

TEST_P(KernelImplementationTest, HaleviShoup8x4) {
  // Original matrix (8x4):
  // [ 0,  1,  2,  3]
  // [ 4,  5,  6,  7]
  // [ 8,  9, 10, 11]
  // [12, 13, 14, 15]
  // [16, 17, 18, 19]
  // [20, 21, 22, 23]
  // [24, 25, 26, 27]
  // [28, 29, 30, 31]
  //
  // Pre-packed diagonally:
  // diag_0 = (0, 5, 10, 15, 16, 21, 26, 31)
  // diag_1 = (1, 6, 11, 12, 17, 22, 27, 28)
  // diag_2 = (2, 7, 8, 13, 18, 23, 24, 29)
  // diag_3 = (3, 4, 9, 14, 19, 20, 25, 30)
  std::vector<std::vector<int>> matrix = {{0, 5, 10, 15, 16, 21, 26, 31},
                                          {1, 6, 11, 12, 17, 22, 27, 28},
                                          {2, 7, 8, 13, 18, 23, 24, 29},
                                          {3, 4, 9, 14, 19, 20, 25, 30}};
  // Replicated vector {3, 4, 5, 6, 3, 4, 5, 6}
  std::vector<int> vector = {3, 4, 5, 6, 3, 4, 5, 6};
  std::vector<int> expected = {32, 104, 176, 248, 320, 392, 464, 536};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {8, 4}, DagType::intTensor(32, {8}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  EXPECT_EQ(actual, expected);
}

TEST_P(KernelImplementationTest, HaleviShoup5x3) {
  // Original matrix (5x3):
  // [ 0,  1,  2]
  // [ 3,  4,  5]
  // [ 6,  7,  8]
  // [ 9, 10, 11]
  // [12, 13, 14]
  //
  // Padded to 8x4 and packed diagonally:
  // diag_0 = (0, 4, 8, 0, 12, 0, 0, 0)
  // diag_1 = (1, 5, 0, 9, 13, 0, 0, 0)
  // diag_2 = (2, 0, 6, 10, 14, 0, 0, 0)
  // diag_3 = (0, 3, 7, 11, 0, 0, 0, 0)
  std::vector<std::vector<int>> matrix = {{0, 4, 8, 0, 12, 0, 0, 0},
                                          {1, 5, 0, 9, 13, 0, 0, 0},
                                          {2, 0, 6, 10, 14, 0, 0, 0},
                                          {0, 3, 7, 11, 0, 0, 0, 0}};
  // Padded and replicated vector {1, 2, 3, 0, 1, 2, 3, 0}
  std::vector<int> vector = {1, 2, 3, 0, 1, 2, 3, 0};
  std::vector<int> expected = {8, 26, 44, 62, 80};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {5, 3}, DagType::intTensor(32, {8}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  // The result is of size 8, but we only care about the first 5 elements.
  EXPECT_EQ(std::vector<int>(actual.begin(), actual.begin() + 5), expected);
}

TEST(KernelImplementationTest, TestExtract) {
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<int> expected = {1, 6, 11, 12};
  LiteralValue matrixInput(matrix);

  auto dag = ArithmeticDagNode<LiteralValue>::extract(
      ArithmeticDagNode<LiteralValue>::leaf(matrixInput), 1);
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST(KernelImplementationTest, TestHaleviShoupMatvecWithLayout) {
  MLIRContext context;
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  std::vector<std::vector<int>> matrix = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  auto diagonalLayout = getDiagonalLayoutRelation(
      RankedTensorType::get({4, 4}, mlir::IndexType::get(&context)), 4);
  std::vector<std::vector<int>> diagonalMatrix =
      evaluateLayoutOnMatrix(diagonalLayout, matrix);

  std::vector<int> expected = {14, 38, 62, 86};
  LiteralValue matrixInput = diagonalMatrix;
  LiteralValue vectorInput = vector;

  auto dag =
      implementMatvec(KernelName::MatvecDiagonal, matrixInput, vectorInput);
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST_P(KernelImplementationTest, Test2DConvWithLayout) {
  MLIRContext context;
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, mlir::IndexType::get(&context));
  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, mlir::IndexType::get(&context));

  int numSlots = 16;
  // 3x3 input data, 2x2 filter
  std::vector<std::vector<int>> data = {{1, -1, 0}, {-3, 0, 2}, {8, 9, 1}};
  std::vector<std::vector<int>> matrix = {{1, -1}, {-1, 1}};

  auto dataLayout = getRowMajorLayoutRelation(dataType, numSlots);
  std::vector<std::vector<int>> packedData =
      evaluateLayoutOnMatrix(dataLayout, data);

  auto matrixLayout = getConvFilterDiagonalizedRelation(filterType, dataType,
                                                        /*padding=*/0, numSlots)
                          .value();
  std::vector<std::vector<int>> packedMatrix =
      evaluateLayoutOnMatrix(matrixLayout, matrix);
  RankedTensorType expandedMatrixType =
      get2dConvFilterExpandedType(filterType, dataType, /*padding=*/0);

  std::vector<int> expected = {5, 1, -2, -10};
  LiteralValue matrixInput = packedMatrix;
  LiteralValue vectorInput = packedData[0];

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, expandedMatrixType.getShape(),
      DagType::intTensor(32, {numSlots}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  // Result is a 2x2 tensor repeated row-major in a tensor of size 16.
  std::vector<int> actualVector = std::get<std::vector<int>>(actual.get());
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
      RankedTensorType::get({m, n}, mlir::IndexType::get(&context)), numSlots);
  auto packedA = evaluateLayoutOnMatrix(layoutA, matrixA);

  auto layoutB = getBicyclicLayoutRelation(
      RankedTensorType::get({n, p}, mlir::IndexType::get(&context)), numSlots);
  auto packedB = evaluateLayoutOnMatrix(layoutB, matrixB);

  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  auto dag = implementBicyclicMatmul(packedAValue, packedBValue, m, n, p,
                                     DagType::intTensor(32, {numSlots}));
  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  auto resultLayout = getBicyclicLayoutRelation(
      RankedTensorType::get({m, p}, mlir::IndexType::get(&context)), numSlots);
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
  int numSlots = m * n * p;

  SymbolicValue packedAValue({m, n}, true);
  SymbolicValue packedBValue({n, p}, true);
  auto dag = implementBicyclicMatmul(packedAValue, packedBValue, m, n, p,
                                     DagType::intTensor(32, {numSlots}));

  RotationCountVisitor rotationCounter;
  int64_t rotationCount = rotationCounter.process(dag);

  // 124 + 2*math.sqrt(124) = 146
  EXPECT_EQ(rotationCount, 146);
}

TEST(KernelImplementationTest, TricyclicBatchMatmul) {
  MLIRContext context;

  // Small example: h=2, m=3, n=5, p=2
  int64_t h = 2;
  int64_t m = 3;
  int64_t n = 5;
  int64_t p = 2;
  int64_t numSlots = h * m * n;

  // Build batch tensors A (h x m x n) and B (h x n x p) with deterministic
  // values
  std::vector<std::vector<std::vector<int>>> A(
      h, std::vector<std::vector<int>>(m, std::vector<int>(n, 0)));
  std::vector<std::vector<std::vector<int>>> B(
      h, std::vector<std::vector<int>>(n, std::vector<int>(p, 0)));

  int val = 1;
  for (int ih = 0; ih < h; ++ih)
    for (int im = 0; im < m; ++im)
      for (int in = 0; in < n; ++in) A[ih][im][in] = val++;
  val = 101;
  for (int ih = 0; ih < h; ++ih)
    for (int in = 0; in < n; ++in)
      for (int ip = 0; ip < p; ++ip) B[ih][in][ip] = val++;

  // Pack using the tricyclic layout relation.
  RankedTensorType typeA =
      RankedTensorType::get({h, m, n}, mlir::IndexType::get(&context));
  RankedTensorType typeB =
      RankedTensorType::get({h, n, p}, mlir::IndexType::get(&context));

  auto layoutA = getTricyclicLayoutRelation(typeA, numSlots);
  auto packedA = evaluateLayoutOnTensor(layoutA, A);

  auto layoutB = getTricyclicLayoutRelation(typeB, numSlots);
  auto packedB = evaluateLayoutOnTensor(layoutB, B);

  // Use ciphertext 0's slot vectors as literal inputs.
  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  // Generate kernel DAG and evaluate
  auto dag =
      implementTricyclicBatchMatmul(packedAValue, packedBValue, h, m, n, p,
                                    DagType::intTensor(32, {numSlots}));
  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  // Compute expected packed φ(Z) via evaluateLayoutOnTensor for the cleartext
  // result. First compute plain CPU batched matmul result (h x m x p).
  std::vector<std::vector<std::vector<int>>> expectedTensor(
      h, std::vector<std::vector<int>>(m, std::vector<int>(p, 0)));
  for (int ih = 0; ih < h; ++ih) {
    for (int im = 0; im < m; ++im) {
      for (int ip = 0; ip < p; ++ip) {
        int sum = 0;
        for (int ic = 0; ic < n; ++ic) {
          sum += A[ih][im][ic] * B[ih][ic][ip];
        }
        expectedTensor[ih][im][ip] = sum;
      }
    }
  }

  RankedTensorType resultType =
      RankedTensorType::get({h, m, p}, mlir::IndexType::get(&context));
  auto resultLayout = getTricyclicLayoutRelation(resultType, numSlots);
  auto expectedPacked = evaluateLayoutOnTensor(resultLayout, expectedTensor);

  // expectedPacked[0] is the expected slot vector for ciphertext 0.
  std::vector<int> expVec = expectedPacked[0];

  // Final check: compare expected packed φ(Z) vector with kernel output.
  EXPECT_EQ(expVec, resultVec);
}

TEST_P(KernelImplementationTest, TestConv2dNchwFchwStride2) {
  MLIRContext context;
  RankedTensorType dataType =
      RankedTensorType::get({1, 1, 4, 4}, mlir::IndexType::get(&context));
  RankedTensorType filterType =
      RankedTensorType::get({4, 1, 2, 2}, mlir::IndexType::get(&context));

  int numSlots = 16;
  // 1x1x4x4 input data, 4x1x2x2 filter
  tensor4d data = {
      {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}}}};
  tensor4d filter = {{{{7, 0}, {5, 4}}},
                     {{{9, 4}, {0, 3}}},
                     {{{7, 8}, {8, 6}}},
                     {{{6, 0}, {5, 4}}}};

  auto dataLayout = getRowMajorLayoutRelation(dataType, numSlots);
  std::vector<std::vector<int>> packedData =
      evaluateLayout(dataLayout, getDataValueFn(data));

  SmallVector<int64_t> strides = {2, 2};
  auto filterLayout = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, 0, numSlots,
      /*interchangeRows=*/std::get<1>(GetParam()));
  ASSERT_TRUE(succeeded(filterLayout));
  std::function<int(const std::vector<int64_t>&)> getFilterValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };
  std::vector<std::vector<int>> packedFilter =
      evaluateLayout(filterLayout.value(), getFilterValueFn);
  auto expandedFilterShape =
      get2dConvChwFchwFilterExpandedType(filterType, dataType, 0, strides);

  // The expected result is a 1x4x2x2:
  tensor4d expected = {{{{40, 72}, {168, 200}},
                        {{19, 51}, {147, 179}},
                        {{70, 128}, {302, 360}},
                        {{40, 70}, {160, 190}}}};

  LiteralValue matrixInput = packedFilter;
  LiteralValue vectorInput = packedData[0];

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, expandedFilterShape.getShape(),
      DagType::intTensor(32, {numSlots}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  auto actual4d = std::get<std::vector<int>>(actual.get());

  RankedTensorType outputType =
      RankedTensorType::get({1, 4, 2, 2}, mlir::IndexType::get(&context));
  auto resultLayout =
      get2dConvResultRelation(outputType, strides, 0, numSlots,
                              /*interchangeRows=*/std::get<1>(GetParam()));

  auto actualUnpacked =
      unpackLayoutTo4DTensor<int>(resultLayout, {actual4d}, {1, 4, 2, 2});

  // Result is 4 2x2 tensors with a row-major layout.
  EXPECT_EQ(actualUnpacked, expected);
}

TEST_P(KernelImplementationTest,
       TestConv2dNchwFchwStride2InterchangedLargeSlots) {
  MLIRContext context;
  RankedTensorType dataType =
      RankedTensorType::get({1, 1, 4, 4}, mlir::IndexType::get(&context));
  RankedTensorType filterType =
      RankedTensorType::get({4, 1, 2, 2}, mlir::IndexType::get(&context));

  int numSlots = 128;
  // 1x1x4x4 input data, 4x1x2x2 filter
  tensor4d data = {
      {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}}}};
  tensor4d filter = {{{{7, 0}, {5, 4}}},
                     {{{9, 4}, {0, 3}}},
                     {{{7, 8}, {8, 6}}},
                     {{{6, 0}, {5, 4}}}};

  auto dataLayout = getRowMajorLayoutRelation(dataType, numSlots);
  std::vector<std::vector<int>> packedData =
      evaluateLayout(dataLayout, getDataValueFn(data));

  SmallVector<int64_t> strides = {2, 2};
  auto filterLayout = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, 0, numSlots, /*interchangeRows=*/true);
  ASSERT_TRUE(succeeded(filterLayout));
  std::function<int(const std::vector<int64_t>&)> getFilterValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };
  std::vector<std::vector<int>> packedFilter =
      evaluateLayout(filterLayout.value(), getFilterValueFn);
  auto expandedFilterShape =
      get2dConvChwFchwFilterExpandedType(filterType, dataType, 0, strides);

  // The expected result is a 1x4x2x2:
  tensor4d expected = {{{{40, 72}, {168, 200}},
                        {{19, 51}, {147, 179}},
                        {{70, 128}, {302, 360}},
                        {{40, 70}, {160, 190}}}};

  LiteralValue matrixInput = packedFilter;
  LiteralValue vectorInput = packedData[0];

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, expandedFilterShape.getShape(),
      DagType::intTensor(32, {numSlots}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  auto actual4d = std::get<std::vector<int>>(actual.get());

  RankedTensorType outputType =
      RankedTensorType::get({1, 4, 2, 2}, mlir::IndexType::get(&context));
  auto resultLayout = get2dConvResultRelation(outputType, strides, 0, numSlots,
                                              /*interchangeRows=*/true);

  auto actualUnpacked =
      unpackLayoutTo4DTensor<int>(resultLayout, {actual4d}, {1, 4, 2, 2});

  // Result is 4 2x2 tensors with a row-major layout.
  EXPECT_EQ(actualUnpacked, expected);
}

TEST_P(KernelImplementationTest, TestConv2dNchwFchwOrionFigure4) {
  MLIRContext context;
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 3, 3}, mlir::IndexType::get(&context));
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 3, 3}, mlir::IndexType::get(&context));

  int numSlots = 32;

  // Initialize data with i % 10
  tensor4d data(
      1, std::vector<std::vector<std::vector<int>>>(
             2, std::vector<std::vector<int>>(3, std::vector<int>(3, 0))));
  for (int f = 0; f < 1; ++f) {
    for (int c = 0; c < 2; ++c) {
      for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
          data[f][c][h][w] = 1 + 3 * h + w;
        }
      }
    }
  }

  // Initialize filter for average pooling: f=c is 1 (using int for test),
  // others 0 Note: we use 1 instead of 0.25 because LiteralValue uses int here.
  // We'll divide by 4 manually in expectation if needed, or just test sum
  // pooling. The MLIR has a separate division op, so let's test sum pooling
  // here.
  tensor4d filter(
      2, std::vector<std::vector<std::vector<int>>>(
             2, std::vector<std::vector<int>>(3, std::vector<int>(3, 0))));
  for (int f = 0; f < 2; ++f) {
    for (int c = 0; c < 2; ++c) {
      for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
          filter[f][c][h][w] = 1 + 3 * h + w;
        }
      }
    }
  }

  auto dataLayout = getRowMajorLayoutRelation(dataType, numSlots);
  std::vector<std::vector<int>> packedData =
      evaluateLayout(dataLayout, getDataValueFn(data));

  SmallVector<int64_t> strides = {1, 1};
  int64_t padding = 1;
  auto filterLayout = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, numSlots,
      /*interchangeRows=*/false);
  ASSERT_TRUE(succeeded(filterLayout));
  std::function<int(const std::vector<int64_t>&)> getFilterValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };
  std::vector<std::vector<int>> packedFilter =
      evaluateLayout(filterLayout.value(), getFilterValueFn);

  auto expandedFilterShape = get2dConvChwFchwFilterExpandedType(
      filterType, dataType, padding, strides);

  // Compute expected 1x2x3x3 2d conv with padding 1
  tensor4d expected(
      1, std::vector<std::vector<std::vector<int>>>(
             2, std::vector<std::vector<int>>(3, std::vector<int>(3, 0))));
  for (int f = 0; f < 2; ++f) {
    for (int ho = 0; ho < 3; ++ho) {
      for (int wo = 0; wo < 3; ++wo) {
        int sum = 0;
        for (int c = 0; c < 2; ++c) {
          for (int hi = 0; hi < 3; ++hi) {
            for (int wi = 0; wi < 3; ++wi) {
              int h = ho + hi - padding;
              int w = wo + wi - padding;
              if (h >= 0 && h < 3 && w >= 0 && w < 3) {
                sum += data[0][c][h][w] * filter[f][c][hi][wi];
              }
            }
          }
        }
        expected[0][f][ho][wo] = sum;
      }
    }
  }

  LiteralValue matrixInput = packedFilter;
  LiteralValue vectorInput = packedData[0];

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, expandedFilterShape.getShape(),
      DagType::intTensor(32, {numSlots}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  auto actualVector = std::get<std::vector<int>>(actual.get());

  RankedTensorType outputType =
      RankedTensorType::get({1, 2, 3, 3}, mlir::IndexType::get(&context));
  auto resultLayout =
      get2dConvResultRelation(outputType, strides, padding, numSlots,
                              /*interchangeRows=*/false);

  auto actualUnpacked =
      unpackLayoutTo4DTensor<int>(resultLayout, {actualVector}, {1, 2, 3, 3});

  EXPECT_EQ(actualUnpacked, expected);
}

TEST_P(KernelImplementationTest, TestConv2dNchwFchwStride2MultiInput) {
  MLIRContext context;
  // Use dimensions that match the pooling test:
  // Input: 1x4x6x6
  // Filter: 4x4x2x2 (independent channel pooling, so f=c has 0.25)
  // Output: 1x4x3x3
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 6, 6}, mlir::IndexType::get(&context));
  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2, 2}, mlir::IndexType::get(&context));

  int numSlots = 1024;

  // Initialize data with i % 10
  tensor4d data(
      1, std::vector<std::vector<std::vector<int>>>(
             4, std::vector<std::vector<int>>(6, std::vector<int>(6, 0))));
  int val = 0;
  for (int c = 0; c < 4; ++c) {
    for (int h = 0; h < 6; ++h) {
      for (int w = 0; w < 6; ++w) {
        data[0][c][h][w] = val % 10;
        val++;
      }
    }
  }

  // Initialize filter for average pooling: f=c is 1 (using int for test),
  // others 0 Note: we use 1 instead of 0.25 because LiteralValue uses int here.
  // We'll divide by 4 manually in expectation if needed, or just test sum
  // pooling. The MLIR has a separate division op, so let's test sum pooling
  // here.
  tensor4d filter(
      4, std::vector<std::vector<std::vector<int>>>(
             4, std::vector<std::vector<int>>(2, std::vector<int>(2, 0))));
  for (int f = 0; f < 4; ++f) {
    for (int c = 0; c < 4; ++c) {
      if (f == c) {
        for (int h = 0; h < 2; ++h) {
          for (int w = 0; w < 2; ++w) {
            filter[f][c][h][w] = 1;
          }
        }
      }
    }
  }

  auto dataLayout = getRowMajorLayoutRelation(dataType, numSlots);
  std::vector<std::vector<int>> packedData =
      evaluateLayout(dataLayout, getDataValueFn(data));

  SmallVector<int64_t> strides = {2, 2};
  auto filterLayout = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, 0, numSlots,
      /*interchangeRows=*/std::get<1>(GetParam()));
  ASSERT_TRUE(succeeded(filterLayout));
  std::function<int(const std::vector<int64_t>&)> getFilterValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };
  std::vector<std::vector<int>> packedFilter =
      evaluateLayout(filterLayout.value(), getFilterValueFn);
  auto expandedFilterShape =
      get2dConvChwFchwFilterExpandedType(filterType, dataType, 0, strides);

  // Compute expected 1x4x3x3 sum pooling
  tensor4d expected(
      1, std::vector<std::vector<std::vector<int>>>(
             4, std::vector<std::vector<int>>(3, std::vector<int>(3, 0))));
  for (int f = 0; f < 4; ++f) {
    for (int ho = 0; ho < 3; ++ho) {
      for (int wo = 0; wo < 3; ++wo) {
        int sum = 0;
        for (int c = 0; c < 4; ++c) {
          for (int hi = 0; hi < 2; ++hi) {
            for (int wi = 0; wi < 2; ++wi) {
              sum +=
                  data[0][c][ho * 2 + hi][wo * 2 + wi] * filter[f][c][hi][wi];
            }
          }
        }
        expected[0][f][ho][wo] = sum;
      }
    }
  }

  LiteralValue matrixInput = packedFilter;
  LiteralValue vectorInput = packedData[0];

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, expandedFilterShape.getShape(),
      DagType::intTensor(32, {numSlots}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  auto actualVector = std::get<std::vector<int>>(actual.get());

  RankedTensorType outputType =
      RankedTensorType::get({1, 4, 3, 3}, mlir::IndexType::get(&context));
  auto resultLayout =
      get2dConvResultRelation(outputType, strides, 0, numSlots,
                              /*interchangeRows=*/std::get<1>(GetParam()));

  auto actualUnpacked =
      unpackLayoutTo4DTensor<int>(resultLayout, {actualVector}, {1, 4, 3, 3});

  EXPECT_EQ(actualUnpacked, expected);
}

INSTANTIATE_TEST_SUITE_P(WithAndWithoutRolledSuite, KernelImplementationTest,
                         testing::Combine(testing::Values(false, true),
                                          testing::Values(false, true)));

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
