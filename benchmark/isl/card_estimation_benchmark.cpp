#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Utils/Layout/IslConversion.h"

// ISL
#include "include/isl/ctx.h"                                        // from @isl
#include "include/isl/map.h"                                        // from @isl
#include "include/isl/map_type.h"                                   // from @isl
#include "include/isl/set.h"                                        // from @isl
#include "include/isl/val.h"                                        // from @isl
#include "include/isl/val_type.h"                                   // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

static void BM_CardEstimation_helper(benchmark::State& state,
                                     const std::string& relation_str) {
  FailureOr<presburger::IntegerRelation> relation_or =
      getIntegerRelationFromIslStr(relation_str);
  if (failed(relation_or)) {
    state.SkipWithError("Failed to parse relation string");
    return;
  }
  const presburger::IntegerRelation& relation = *relation_or;

  isl_ctx* ctx = isl_ctx_alloc();
  for (auto _ : state) {
    isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);
    isl_set* set = isl_set_from_basic_set(isl_basic_map_wrap(bmap));
    isl_val* card = isl_set_count_val(set);
    int64_t count = isl_val_get_num_si(card);
    isl_val_free(card);
    isl_set_free(set);
    benchmark::DoNotOptimize(count);
  }
  isl_ctx_free(ctx);
}

static void BM_CardEstimation_Relation1(benchmark::State& state) {
  BM_CardEstimation_helper(state, kRelation1);
}
BENCHMARK(BM_CardEstimation_Relation1)->Unit(benchmark::kSecond);

static void BM_CardEstimation_Relation2(benchmark::State& state) {
  BM_CardEstimation_helper(state, kRelation2);
}
BENCHMARK(BM_CardEstimation_Relation2)->Unit(benchmark::kSecond);

static void BM_CardEstimation_Relation3(benchmark::State& state) {
  BM_CardEstimation_helper(state, kRelation3);
}
BENCHMARK(BM_CardEstimation_Relation3)->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
