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
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace kernel {
namespace {

void tricyclicBatchMatmulMatchesNaive(
    const std::tuple<int64_t, int64_t, int64_t, int64_t, std::vector<int>,
                     std::vector<int>>& args) {
  const auto& [h, m, n, p, vecA, vecB] = args;

  if (std::gcd(h, m) != 1 || std::gcd(h, n) != 1 || std::gcd(h, p) != 1 ||
      std::gcd(m, n) != 1 || std::gcd(m, p) != 1 || std::gcd(n, p) != 1) {
    return;
  }

  MLIRContext context;
  int64_t minSlots = h * (m * n + n * p + m * p);
  int64_t numSlots = 1;
  while (numSlots <= minSlots) numSlots *= 2;

  RankedTensorType typeA =
      RankedTensorType::get({h, m, n}, mlir::IndexType::get(&context));
  RankedTensorType typeB =
      RankedTensorType::get({h, n, p}, mlir::IndexType::get(&context));

  auto layoutA = getTricyclicLayoutRelation(typeA, numSlots);
  auto packedA =
      evaluateLayout<int>(layoutA, [&](const std::vector<int64_t>& pt) {
        return vecA[pt[0] * m * n + pt[1] * n + pt[2]];
      });

  auto layoutB = getTricyclicLayoutRelation(typeB, numSlots);
  auto packedB =
      evaluateLayout<int>(layoutB, [&](const std::vector<int64_t>& pt) {
        return vecB[pt[0] * n * p + pt[1] * p + pt[2]];
      });

  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  auto dag =
      implementTricyclicBatchMatmul(packedAValue, packedBValue, h, m, n, p,
                                    DagType::intTensor(32, {numSlots}));
  LiteralValue result = evalKernel(dag)[0];
  auto actualVec = std::get<std::vector<int>>(result.get());

  RankedTensorType resultType =
      RankedTensorType::get({h, m, p}, mlir::IndexType::get(&context));
  auto resultLayout = getTricyclicLayoutRelation(resultType, numSlots);
  // Restrict the unpacking to the first output period.
  addBounds(resultLayout,
            resultLayout.getVarKindOffset(presburger::VarKind::Range) + 1, 0,
            h * m * p - 1);
  auto actual =
      unpackLayoutTo3DTensor<int>(resultLayout, {actualVec}, {h, m, p});

  std::vector<std::vector<std::vector<int>>> expected(
      h, std::vector<std::vector<int>>(m, std::vector<int>(p, 0)));
  for (int64_t ih = 0; ih < h; ++ih) {
    for (int64_t im = 0; im < m; ++im) {
      for (int64_t ip = 0; ip < p; ++ip) {
        for (int64_t in = 0; in < n; ++in) {
          expected[ih][im][ip] +=
              vecA[ih * m * n + im * n + in] * vecB[ih * n * p + in * p + ip];
        }
      }
    }
  }

  EXPECT_EQ(expected, actual);
}

auto tricyclicShapeAndTensors() {
  return fuzztest::FlatMap(
      [](int64_t h, int64_t m, int64_t n, int64_t p) {
        return fuzztest::TupleOf(
            fuzztest::Just(h), fuzztest::Just(m), fuzztest::Just(n),
            fuzztest::Just(p),
            /*vecA=*/
            fuzztest::VectorOf(fuzztest::InRange(-10, 10)).WithSize(h * m * n),
            /*vecB=*/
            fuzztest::VectorOf(fuzztest::InRange(-10, 10)).WithSize(h * n * p));
      },
      /*h=*/fuzztest::InRange<int64_t>(1, 8),
      /*m=*/fuzztest::InRange<int64_t>(1, 8),
      /*n=*/fuzztest::InRange<int64_t>(1, 8),
      /*p=*/fuzztest::InRange<int64_t>(1, 8));
}

TEST(TricyclicBatchMatmulFuzzTest, TricyclicBatchMatmulRegression) {
  tricyclicBatchMatmulMatchesNaive(
      {2,
       3,
       5,
       7,
       {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
       {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70}});
}

FUZZ_TEST(TricyclicBatchMatmulFuzzTest, tricyclicBatchMatmulMatchesNaive)
    .WithDomains(tricyclicShapeAndTensors());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
