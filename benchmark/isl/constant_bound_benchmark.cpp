// Compares two ways of computing a constant upper bound on the ct range
// variable of a layout. Layout code calls IntegerRelation::getConstantBound64
// for this in several places (AssignLayout, LayoutPropagation,
// LayoutConversionCost); materializeLayout uses Simplex::computeIntegerBounds.

#include <cstdint>
#include <optional>
#include <string>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "llvm/include/llvm/ADT/DynamicAPInt.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"   // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/Simplex.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;

static void BM_ConstantBound_helper(benchmark::State& state,
                                    const std::string& relation_str) {
  FailureOr<IntegerRelation> relation_or =
      getIntegerRelationFromIslStr(relation_str);
  if (failed(relation_or)) {
    state.SkipWithError("Failed to parse relation string");
    return;
  }
  const IntegerRelation& relation = *relation_or;
  unsigned ctPos = relation.getVarKindOffset(VarKind::Range);

  // Arg 0: IntegerRelation::getConstantBound64, which projects out every other
  // variable with Fourier-Motzkin elimination.
  // Arg 1: Simplex::computeIntegerBounds.
  bool useSimplex = state.range(0) == 1;
  int64_t bound = 0;
  for (auto _ : state) {
    if (useSimplex) {
      presburger::Simplex simplex(relation);
      llvm::SmallVector<llvm::DynamicAPInt> coeffs(relation.getNumVars() + 1,
                                                   llvm::DynamicAPInt(0));
      coeffs[ctPos] = llvm::DynamicAPInt(1);
      auto bounds = simplex.computeIntegerBounds(coeffs);
      bound = bounds.second.isBounded() ? int64_t(*bounds.second) : -1;
    } else {
      std::optional<int64_t> ub =
          relation.getConstantBound64(BoundType::UB, ctPos);
      bound = ub.value_or(-1);
    }
    benchmark::DoNotOptimize(bound);
  }
  state.counters["ct_ub"] = bound;
}

static void BM_ConstantBound_LolaGapFilter(benchmark::State& state) {
  BM_ConstantBound_helper(state, kLolaGapFilterRelation);
}
// Arg 0 runs out of memory: with an 18 GB limit it aborts after about 140 s.
BENCHMARK(BM_ConstantBound_LolaGapFilter)
    // ->Arg(0)
    ->Arg(1)
    ->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
