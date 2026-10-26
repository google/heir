// Times isl_map_compute_divs, which isl runs before scanning, counting or
// comparing a map whose existentials have no explicit division form. It can
// fall into parametric integer programming, so isl_set_foreach_point,
// isl_set_count_val and isl_map_is_equal all inherit its cost. On kLolaLayout12
// it does not finish within 10 minutes.

#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Utils/Layout/IslConversion.h"

// ISL
#include "include/isl/ctx.h"                                        // from @isl
#include "include/isl/map.h"                                        // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

static void BM_ComputeDivs_helper(benchmark::State& state,
                                  const std::string& relation_str) {
  FailureOr<presburger::IntegerRelation> relation_or =
      getIntegerRelationFromIslStr(relation_str);
  if (failed(relation_or)) {
    state.SkipWithError("Failed to parse relation string");
    return;
  }
  const presburger::IntegerRelation& relation = *relation_or;

  isl_ctx* ctx = isl_ctx_alloc();
  int64_t pieces = 0;
  for (auto _ : state) {
    isl_map* map =
        isl_map_from_basic_map(convertRelationToBasicMap(relation, ctx));
    map = isl_map_compute_divs(map);
    pieces = isl_map_n_basic_map(map);
    isl_map_free(map);
    benchmark::DoNotOptimize(pieces);
  }
  state.counters["pieces"] = pieces;
  isl_ctx_free(ctx);
}

static void BM_ComputeDivs_DscnnRemap4(benchmark::State& state) {
  BM_ComputeDivs_helper(state, kDscnnRemap4Relation);
}
BENCHMARK(BM_ComputeDivs_DscnnRemap4)->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
