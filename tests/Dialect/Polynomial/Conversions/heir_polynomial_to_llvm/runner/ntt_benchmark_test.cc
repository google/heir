// Block clang-format from reordering
// clang-format off
#include "benchmark/benchmark.h"  // from @google_benchmark
// clang-format on
#include "tests/Examples/benchmark/Memref.h"

namespace heir {
namespace {

using ::heir::test::Memref;

extern "C" void _mlir_ciface_input_generation(Memref* output);
extern "C" void _mlir_ciface_ntt(Memref* output, Memref* input);
extern "C" void _mlir_ciface_intt(Memref* output, Memref* input);

void BM_ntt_benchmark(benchmark::State& state) {
  Memref input(1, 65536, 0);
  _mlir_ciface_input_generation(&input);

  Memref ntt(1, 65536, 0);
  while (state.KeepRunning()) {
    _mlir_ciface_ntt(&ntt, &input);
  }

  Memref intt(1, 65536, 0);
  _mlir_ciface_intt(&intt, &ntt);
}

BENCHMARK(BM_ntt_benchmark);

void BM_intt_benchmark(benchmark::State& state) {
  Memref input(1, 65536, 0);
  _mlir_ciface_input_generation(&input);

  Memref ntt(1, 65536, 0);
  _mlir_ciface_ntt(&ntt, &input);

  Memref intt(1, 65536, 0);
  while (state.KeepRunning()) {
    _mlir_ciface_intt(&intt, &ntt);
  }
}

BENCHMARK(BM_intt_benchmark);

}  // namespace
}  // namespace heir
