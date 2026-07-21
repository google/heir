#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

static void BM_GetCtComplementPoints_helper(benchmark::State& state,
                                            const std::string& relation_str,
                                            int64_t numCts) {
  FailureOr<presburger::IntegerRelation> relation_or =
      getIntegerRelationFromIslStr(relation_str);
  if (failed(relation_or)) {
    state.SkipWithError("Failed to parse relation string");
    return;
  }
  const presburger::IntegerRelation& relation = *relation_or;

  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({numCts, 4096}, IndexType::get(&context));

  PointCollector collector;
  for (auto _ : state) {
    collector.points.clear();
    getCtComplementPoints(relation, collector, type);
    benchmark::DoNotOptimize(collector.points);
  }
}

static void BM_GetCtComplementPoints_Layout4(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout4Relation, 2048);
}
BENCHMARK(BM_GetCtComplementPoints_Layout4)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout8(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout8Relation, 2048);
}
BENCHMARK(BM_GetCtComplementPoints_Layout8)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout13(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout13Relation, 2048);
}
BENCHMARK(BM_GetCtComplementPoints_Layout13)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout17(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout17Relation, 2048);
}
BENCHMARK(BM_GetCtComplementPoints_Layout17)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout19(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout19Relation, 1024);
}
BENCHMARK(BM_GetCtComplementPoints_Layout19)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout24(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout24Relation, 1024);
}
BENCHMARK(BM_GetCtComplementPoints_Layout24)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout27(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout27Relation, 1024);
}
BENCHMARK(BM_GetCtComplementPoints_Layout27)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout29(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout29Relation, 1024);
}
BENCHMARK(BM_GetCtComplementPoints_Layout29)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout34(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout34Relation, 1024);
}
BENCHMARK(BM_GetCtComplementPoints_Layout34)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout37(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout37Relation, 1024);
}
BENCHMARK(BM_GetCtComplementPoints_Layout37)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_Layout39(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayout39Relation, 64);
}
BENCHMARK(BM_GetCtComplementPoints_Layout39)->Unit(benchmark::kSecond);

static void BM_GetCtComplementPoints_LayoutPooling1(benchmark::State& state) {
  BM_GetCtComplementPoints_helper(state, kLayoutPooling1Relation, 8192);
}
BENCHMARK(BM_GetCtComplementPoints_LayoutPooling1)->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
