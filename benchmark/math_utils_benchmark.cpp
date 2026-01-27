#include <cstdint>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

static void BM_findPrimitive2nthRoot(benchmark::State& state) {
  uint64_t q_val = state.range(0);
  uint64_t n = state.range(1);
  APInt q(64, q_val);

  for (auto _ : state) {
    auto root = findPrimitive2nthRoot(q, n);
    benchmark::DoNotOptimize(root);
  }
}

// q = 65537, n = 1024
BENCHMARK(BM_findPrimitive2nthRoot)
    ->Args({65537, 1024})
    ->Unit(benchmark::kMicrosecond);
// q = 114689, n = 1024
BENCHMARK(BM_findPrimitive2nthRoot)
    ->Args({114689, 1024})
    ->Unit(benchmark::kMicrosecond);
// q = 2147565569, n = 8192
BENCHMARK(BM_findPrimitive2nthRoot)
    ->Args({2147565569, 8192})
    ->Unit(benchmark::kMicrosecond);
// A larger prime: 2^32 + 15 * 2^27 + 1 = 4294967297 + 2013265920 + 1? No.
// Let's use 3221225473 from MathUtilsTest.cpp
BENCHMARK(BM_findPrimitive2nthRoot)
    ->Args({3221225473, 8192})
    ->Unit(benchmark::kMicrosecond);
// q = 1152921504606846977, n = 65536
BENCHMARK(BM_findPrimitive2nthRoot)
    ->Args({1152921504606846977, 65536})
    ->Unit(benchmark::kMicrosecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
