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

std::vector<std::vector<int>> runBicyclicMatmul(const std::vector<int>& vecA,
                                                const std::vector<int>& vecB,
                                                int64_t m, int64_t n,
                                                int64_t p) {
  MLIRContext context;
  int64_t numSlots = m * n * p;

  auto layoutA = getBicyclicLayoutRelation(
      RankedTensorType::get({m, n}, mlir::IndexType::get(&context)), numSlots);
  auto packedA = evaluateLayout<int>(
      layoutA,
      [&](const std::vector<int64_t>& pt) { return vecA[pt[0] * n + pt[1]]; });

  auto layoutB = getBicyclicLayoutRelation(
      RankedTensorType::get({n, p}, mlir::IndexType::get(&context)), numSlots);
  auto packedB = evaluateLayout<int>(
      layoutB,
      [&](const std::vector<int64_t>& pt) { return vecB[pt[0] * p + pt[1]]; });

  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  auto dag = implementBicyclicMatmul(packedAValue, packedBValue, m, n, p,
                                     DagType::intTensor(32, {numSlots}));
  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  auto resultLayout = getBicyclicLayoutRelation(
      RankedTensorType::get({m, p}, mlir::IndexType::get(&context)), numSlots);
  return unpackLayoutToMatrix<int>(resultLayout, {resultVec}, {m, p});
}

void bicyclicMatmulMatchesNaive(
    const std::tuple<int64_t, int64_t, int64_t, std::vector<int>,
                     std::vector<int>>& args) {
  const auto& [m, n, p, vecA, vecB] = args;

  if (std::gcd(m, n) != 1 || std::gcd(n, p) != 1 || std::gcd(m, p) != 1) return;

  std::vector<std::vector<int>> expected(m, std::vector<int>(p, 0));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < p; ++j) {
      for (int64_t k = 0; k < n; ++k) {
        expected[i][j] += vecA[i * n + k] * vecB[k * p + j];
      }
    }
  }

  std::vector<std::vector<int>> actual = runBicyclicMatmul(vecA, vecB, m, n, p);

  EXPECT_EQ(expected, actual);
}

auto shapeAndMatrices() {
  return fuzztest::FlatMap(
      [](int64_t m, int64_t n, int64_t p) {
        return fuzztest::TupleOf(
            fuzztest::Just(m), fuzztest::Just(n), fuzztest::Just(p),
            /*vecA=*/
            fuzztest::VectorOf(fuzztest::InRange(-100, 100)).WithSize(m * n),
            /*vecB=*/
            fuzztest::VectorOf(fuzztest::InRange(-100, 100)).WithSize(n * p));
      },
      /*m=*/fuzztest::InRange<int64_t>(1, 16),
      /*n=*/fuzztest::InRange<int64_t>(1, 16),
      /*p=*/fuzztest::InRange<int64_t>(1, 16));
}

TEST(BicyclicMatmulFuzzTest, BicyclicMatmulRegression) {
  bicyclicMatmulMatchesNaive(
      {3,
       5,
       2,
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
}

FUZZ_TEST(BicyclicMatmulFuzzTest, bicyclicMatmulMatchesNaive)
    .WithDomains(shapeAndMatrices());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
