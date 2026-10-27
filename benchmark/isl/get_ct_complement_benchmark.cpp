#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"

// ISL
#include "include/isl/ctx.h"                                        // from @isl
#include "include/isl/map.h"                                        // from @isl
#include "include/isl/point.h"                                      // from @isl
#include "include/isl/set.h"                                        // from @isl
#include "include/isl/val.h"                                        // from @isl
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

// Earlier ways of finding which cts a layout uses, kept for the isl calls they
// make.

// One isl_basic_map_fix_val on the ct output plus isl_basic_map_is_empty per
// ct.
static void BM_PerCtEmptiness_helper(benchmark::State& state,
                                     const std::string& relation_str,
                                     int64_t numCts) {
  FailureOr<presburger::IntegerRelation> relation_or =
      getIntegerRelationFromIslStr(relation_str);
  if (failed(relation_or)) {
    state.SkipWithError("Failed to parse relation string");
    return;
  }
  const presburger::IntegerRelation& relation = *relation_or;

  isl_ctx* ctx = isl_ctx_alloc();
  std::vector<bool> seen;
  for (auto _ : state) {
    seen.assign(numCts, false);
    isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);
    for (int64_t ct = 0; ct < numCts; ++ct) {
      isl_basic_map* fixed =
          isl_basic_map_fix_val(isl_basic_map_copy(bmap), isl_dim_out, 0,
                                isl_val_int_from_si(ctx, ct));
      if (isl_basic_map_is_empty(fixed) == isl_bool_false) seen[ct] = true;
      isl_basic_map_free(fixed);
    }
    isl_basic_map_free(bmap);
    benchmark::DoNotOptimize(seen);
  }
  isl_ctx_free(ctx);
}

static void BM_PerCtEmptiness_LolaGapFilter(benchmark::State& state) {
  BM_PerCtEmptiness_helper(state, kLolaGapFilterRelation, 1024);
}
BENCHMARK(BM_PerCtEmptiness_LolaGapFilter)->Unit(benchmark::kSecond);

static isl_stat markCt(__isl_take isl_point* pnt, void* user) {
  auto* seen = static_cast<std::vector<bool>*>(user);
  isl_val* ct = isl_point_get_coordinate_val(pnt, isl_dim_set, 0);
  int64_t value = isl_val_get_num_si(ct);
  if (value >= 0 && value < static_cast<int64_t>(seen->size()))
    (*seen)[value] = true;
  isl_val_free(ct);
  isl_point_free(pnt);
  return isl_stat_ok;
}

// isl_set_project_out of the slot from the range, then isl_set_foreach_point
// over the remaining ct set.
static void BM_ProjectOutSlot_helper(benchmark::State& state,
                                     const std::string& relation_str,
                                     int64_t numCts) {
  FailureOr<presburger::IntegerRelation> relation_or =
      getIntegerRelationFromIslStr(relation_str);
  if (failed(relation_or)) {
    state.SkipWithError("Failed to parse relation string");
    return;
  }
  const presburger::IntegerRelation& relation = *relation_or;

  isl_ctx* ctx = isl_ctx_alloc();
  std::vector<bool> seen;
  for (auto _ : state) {
    seen.assign(numCts, false);
    isl_set* ctSet = isl_set_from_basic_set(
        isl_basic_map_range(convertRelationToBasicMap(relation, ctx)));
    ctSet = isl_set_project_out(ctSet, isl_dim_set, 1, 1);
    isl_set_foreach_point(ctSet, markCt, &seen);
    isl_set_free(ctSet);
    benchmark::DoNotOptimize(seen);
  }
  isl_ctx_free(ctx);
}

[[maybe_unused]] static void BM_ProjectOutSlot_Layout19(
    benchmark::State& state) {
  BM_ProjectOutSlot_helper(state, kLayout19Relation, 1024);
}
// Does not finish within 10 minutes.
// BENCHMARK(BM_ProjectOutSlot_Layout19)->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
