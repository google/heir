#include <string>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

static void BM_EnumeratePoints_helper(benchmark::State& state,
                                      const std::string& relation_str) {
  FailureOr<presburger::IntegerRelation> relation_or =
      getIntegerRelationFromIslStr(relation_str);
  if (failed(relation_or)) {
    state.SkipWithError("Failed to parse relation string");
    return;
  }
  const presburger::IntegerRelation& relation = *relation_or;

  PointPairCollector collector(relation.getNumDomainVars(),
                               relation.getNumRangeVars());
  for (auto _ : state) {
    collector.points.clear();
    enumeratePoints(relation, collector);
    benchmark::DoNotOptimize(collector.points);
  }
}

static void BM_EnumeratePoints_Relation1(benchmark::State& state) {
  BM_EnumeratePoints_helper(state, kRelation1);
}
BENCHMARK(BM_EnumeratePoints_Relation1)->Unit(benchmark::kSecond);

static void BM_EnumeratePoints_Relation2(benchmark::State& state) {
  BM_EnumeratePoints_helper(state, kRelation2);
}
BENCHMARK(BM_EnumeratePoints_Relation2)->Unit(benchmark::kSecond);

static void BM_EnumeratePoints_Relation3(benchmark::State& state) {
  BM_EnumeratePoints_helper(state, kRelation3);
}
BENCHMARK(BM_EnumeratePoints_Relation3)->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
