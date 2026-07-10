#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
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

// Naive matvec using diagonalized matrix
std::vector<int> runNaiveMatvec(
    const std::vector<int>& vector,
    const std::vector<std::vector<int>>& diagMatrix) {
  int n = vector.size();
  std::vector<int> result(n, 0);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      result[i] += diagMatrix[k][i] * vector[(i + k) % n];
    }
  }
  return result;
}

void matvecZeroDiagonalsMatchesNaive(
    const std::tuple<std::vector<int>, std::vector<std::vector<int>>,
                     std::vector<int>>& args,
    bool unroll) {
  const auto& [vector, diagMatrix, zeroDiagonals] = args;
  int n = vector.size();
  if (n == 0) return;

  // Prepare zeroDiagonals map
  std::map<int, bool> zeroDiagonalsMap;
  for (int diag : zeroDiagonals) {
    zeroDiagonalsMap[diag] = true;
  }

  // Create a copy of diagMatrix and physically zero out the selected diagonals
  std::vector<std::vector<int>> zeroedDiagMatrix = diagMatrix;
  for (int diag : zeroDiagonals) {
    if (diag >= 0 && diag < n) {
      std::fill(zeroedDiagMatrix[diag].begin(), zeroedDiagMatrix[diag].end(),
                0);
    }
  }

  // Run naive with physically zeroed matrix
  std::vector<int> expected = runNaiveMatvec(vector, zeroedDiagMatrix);

  // Run kernel implementation with zeroDiagonals map
  LiteralValue matrixInput(zeroedDiagMatrix);
  LiteralValue vectorInput(vector);

  auto dag =
      implementHaleviShoup(vectorInput, matrixInput,
                           {static_cast<int64_t>(n), static_cast<int64_t>(n)},
                           DagType::intTensor(32, {static_cast<int64_t>(n)}),
                           zeroDiagonalsMap, unroll);

  std::vector<int> actual =
      std::get<std::vector<int>>(evalKernel(dag)[0].get());

  EXPECT_EQ(expected, actual);
}

FUZZ_TEST(MatvecZeroDiagonalsFuzzTest, matvecZeroDiagonalsMatchesNaive)
    .WithDomains(fuzztest::FlatMap(
                     [](size_t n) {
                       return fuzztest::TupleOf(
                           /*vector=*/fuzztest::VectorOf(
                               fuzztest::InRange(-100, 100))
                               .WithSize(n),
                           /*diagMatrix=*/
                           fuzztest::VectorOf(
                               fuzztest::VectorOf(fuzztest::InRange(-100, 100))
                                   .WithSize(n))
                               .WithSize(n),
                           /*zeroDiagonals=*/
                           fuzztest::VectorOf(fuzztest::InRange<int>(0, n - 1))
                               .WithMinSize(0)
                               .WithMaxSize(n));
                     },
                     /*n=*/fuzztest::InRange<size_t>(1, 16)),
                 fuzztest::Arbitrary<bool>());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
