#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace kernel {
namespace {

std::vector<std::vector<int>> runDiagonalMatmul(bool isCtPt,
                                                const std::vector<int>& vecCt,
                                                const std::vector<int>& vecPt,
                                                int64_t m, int64_t n,
                                                int64_t p) {
  MLIRContext context;
  int64_t numSlots = 2 * m * n * p;

  int64_t rowsCt = isCtPt ? m : n;
  int64_t colsCt = isCtPt ? n : p;

  auto layoutCt = getBicyclicLayoutRelation(
      RankedTensorType::get({rowsCt, colsCt}, mlir::IndexType::get(&context)),
      numSlots);

  auto packedCt =
      evaluateLayout<int>(layoutCt, [&](const std::vector<int64_t>& pt) {
        return vecCt[pt[0] * colsCt + pt[1]];
      });

  int64_t rowsPt = isCtPt ? n : m;
  int64_t colsPt = isCtPt ? p : n;
  int64_t contractionDim = isCtPt ? 0 : 1;
  int64_t stride = isCtPt ? m : p;
  int64_t steps = n;
  int64_t period = isCtPt ? m : p;

  RankedTensorType weightType =
      RankedTensorType::get({rowsPt, colsPt}, mlir::IndexType::get(&context));
  auto layoutPt =
      getBicyclicDiagonalRelation(weightType, contractionDim, stride, numSlots);
  auto packedPt =
      evaluateLayout<int>(layoutPt, [&](const std::vector<int64_t>& pt) {
        return vecPt[pt[0] * colsPt + pt[1]];
      });

  LiteralValue secretVal = packedCt[0];
  LiteralValue plainVal = packedPt;

  auto dag = implementRotateAndReduce(
      secretVal, std::optional<LiteralValue>(plainVal), period, steps,
      DagType::intTensor(32, {numSlots}));

  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  auto resultLayout = getBicyclicLayoutRelation(
      RankedTensorType::get({m, p}, mlir::IndexType::get(&context)), numSlots);

  return unpackLayoutToMatrix<int>(resultLayout, {resultVec}, {m, p});
}

void diagonalMatmulMatchesNaive(
    const std::tuple<bool, int64_t, int64_t, int64_t, std::vector<int>,
                     std::vector<int>>& args) {
  const auto& [isCtPt, m, n, p, vecCt, vecPt] = args;

  if (std::gcd(m, n) != 1 || std::gcd(n, p) != 1 || std::gcd(m, p) != 1) return;

  std::vector<std::vector<int>> expected(m, std::vector<int>(p, 0));
  if (isCtPt) {
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < p; ++j) {
        for (int64_t k = 0; k < n; ++k) {
          expected[i][j] += vecCt[i * n + k] * vecPt[k * p + j];
        }
      }
    }
  } else {
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < p; ++j) {
        for (int64_t k = 0; k < n; ++k) {
          expected[i][j] += vecPt[i * n + k] * vecCt[k * p + j];
        }
      }
    }
  }

  std::vector<std::vector<int>> actual =
      runDiagonalMatmul(isCtPt, vecCt, vecPt, m, n, p);

  EXPECT_EQ(expected, actual);
}

auto shapeAndMatrices() {
  return fuzztest::FlatMap(
      [](bool isCtPt, int64_t m, int64_t n, int64_t p) {
        int64_t sizeCt = isCtPt ? m * n : n * p;
        int64_t sizePt = isCtPt ? n * p : m * n;
        return fuzztest::TupleOf(
            fuzztest::Just(isCtPt), fuzztest::Just(m), fuzztest::Just(n),
            fuzztest::Just(p),
            /*vecCt=*/
            fuzztest::VectorOf(fuzztest::InRange(-100, 100)).WithSize(sizeCt),
            /*vecPt=*/
            fuzztest::VectorOf(fuzztest::InRange(-100, 100)).WithSize(sizePt));
      },
      /*isCtPt=*/fuzztest::Arbitrary<bool>(),
      /*m=*/fuzztest::InRange<int64_t>(1, 16),
      /*n=*/fuzztest::InRange<int64_t>(1, 16),
      /*p=*/fuzztest::InRange<int64_t>(1, 16));
}

TEST(BicyclicDiagonalMatmulFuzzTest, CtPtRegression) {
  diagonalMatmulMatchesNaive(
      {true,
       3,
       5,
       2,
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
}

TEST(BicyclicDiagonalMatmulFuzzTest, PtCtRegression) {
  diagonalMatmulMatchesNaive(
      {false,
       3,
       5,
       2,
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}});
}

TEST(BicyclicDiagonalMatmulFuzzTest, UnitRowDimRegression) {
  diagonalMatmulMatchesNaive(
      {true, 1, 5, 2, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
}

TEST(BicyclicDiagonalMatmulFuzzTest, UnitContractionDimRegression) {
  diagonalMatmulMatchesNaive({true, 3, 1, 2, {1, 2, 3}, {4, 5}});
}

FUZZ_TEST(BicyclicDiagonalMatmulFuzzTest, diagonalMatmulMatchesNaive)
    .WithDomains(shapeAndMatrices());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
