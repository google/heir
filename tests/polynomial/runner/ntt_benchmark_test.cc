// Block clang-format from reordering
// clang-format off
#include "benchmark/benchmark.h" // from @google_benchmark
#include "gtest/gtest.h" // from @googletest
// clang-format on
#include "tests/benchmark/Memref.h"

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
  for (auto _ : state) {
    _mlir_ciface_ntt(&ntt, &input);
  }

  Memref intt(1, 65536, 0);
  _mlir_ciface_intt(&intt, &ntt);

  for (int i = 0; i < 65526; i++) {
    EXPECT_EQ(intt.get(0, i), input.get(0, i));
  }
}

BENCHMARK(BM_ntt_benchmark);

void BM_intt_benchmark(benchmark::State& state) {
  Memref input(1, 65536, 0);
  _mlir_ciface_input_generation(&input);

  Memref ntt(1, 65536, 0);
  _mlir_ciface_ntt(&ntt, &input);

  Memref intt(1, 65536, 0);
  for (auto _ : state) {
    _mlir_ciface_intt(&intt, &ntt);
  }

  for (int i = 0; i < 65526; i++) {
    EXPECT_EQ(intt.get(0, i), input.get(0, i));
  }
}

BENCHMARK(BM_intt_benchmark);

}  // namespace
}  // namespace heir
