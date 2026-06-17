#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/APInt.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

// Duplicate of the slow path from APIntUtils.cpp
std::vector<APInt> factorize_slow(APInt n) {
  std::vector<APInt> factors;
  if (n.ult(2)) return factors;

  unsigned width = n.getBitWidth();

  APInt d(width, 2);
  while (true) {
    APInt wide_d = d.zext(width * 2);
    if ((wide_d * wide_d).ugt(n.zext(width * 2))) break;

    if (n.urem(d).isZero()) {
      factors.push_back(d);
      while (n.urem(d).isZero()) {
        n = n.udiv(d);
      }
    }
    ++d;
  }
  if (n.ugt(1)) {
    factors.push_back(n);
  }
  return factors;
}

constexpr uint64_t kSmallPrime = 2147483647;             // 2^31 - 1
constexpr uint64_t kMediumComposite = 1000036000099ULL;  // 1000003 * 1000033

void BM_FastPath_SmallPrime(benchmark::State& state) {
  APInt n(64, kSmallPrime);
  for (auto _ : state) {
    benchmark::DoNotOptimize(factorize(n));
  }
}
BENCHMARK(BM_FastPath_SmallPrime);

void BM_SlowPath_SmallPrime(benchmark::State& state) {
  APInt n(64, kSmallPrime);
  for (auto _ : state) {
    benchmark::DoNotOptimize(factorize_slow(n));
  }
}
BENCHMARK(BM_SlowPath_SmallPrime);

void BM_FastPath_MediumComposite(benchmark::State& state) {
  APInt n(64, kMediumComposite);
  for (auto _ : state) {
    benchmark::DoNotOptimize(factorize(n));
  }
}
BENCHMARK(BM_FastPath_MediumComposite);

void BM_SlowPath_MediumComposite(benchmark::State& state) {
  APInt n(64, kMediumComposite);
  for (auto _ : state) {
    benchmark::DoNotOptimize(factorize_slow(n));
  }
}
BENCHMARK(BM_SlowPath_MediumComposite);

}  // namespace
}  // namespace heir
}  // namespace mlir
