// Times isl_set_is_equal of a layout's image set against the full
// [numCts, numSlots] box, with no operation budget. This is how a layout was
// once checked for being dense. On the range of kLayoutPooling1Relation it does
// not finish within 10 minutes.

#include <cstdint>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"

// ISL
#include "include/isl/ctx.h"  // from @isl
#include "include/isl/set.h"  // from @isl
#include "include/isl/val.h"  // from @isl

namespace mlir {
namespace heir {

static bool isDenseBox(isl_set* imageSet, int64_t numCts, int64_t numSlots) {
  isl_ctx* ctx = isl_set_get_ctx(imageSet);
  isl_set* denseSet = isl_set_universe(isl_set_get_space(imageSet));
  denseSet = isl_set_lower_bound_val(denseSet, isl_dim_set, 0,
                                     isl_val_int_from_si(ctx, 0));
  denseSet = isl_set_upper_bound_val(denseSet, isl_dim_set, 0,
                                     isl_val_int_from_si(ctx, numCts - 1));
  denseSet = isl_set_lower_bound_val(denseSet, isl_dim_set, 1,
                                     isl_val_int_from_si(ctx, 0));
  denseSet = isl_set_upper_bound_val(denseSet, isl_dim_set, 1,
                                     isl_val_int_from_si(ctx, numSlots - 1));
  bool result = isl_set_is_equal(imageSet, denseSet) == isl_bool_true;
  isl_set_free(imageSet);
  isl_set_free(denseSet);
  return result;
}

static void BM_DenseCheck_ConvImageSet(benchmark::State& state) {
  isl_ctx* ctx = isl_ctx_alloc();
  bool dense = false;
  for (auto _ : state) {
    isl_set* imageSet = isl_set_read_from_str(ctx, kConvDenseCheckImageSet);
    dense = isDenseBox(imageSet, 2048, 4096);
    benchmark::DoNotOptimize(dense);
  }
  isl_ctx_free(ctx);
}
BENCHMARK(BM_DenseCheck_ConvImageSet)->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
