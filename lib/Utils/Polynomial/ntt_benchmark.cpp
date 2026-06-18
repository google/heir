#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "lib/Utils/MathUtils.h"
#include "lib/Utils/Polynomial/NTT.h"
#include "llvm/include/llvm/ADT/APInt.h"  // from @llvm-project

// OpenFHE includes
#include "src/core/include/lattice/hal/lat-backend.h"    // from @openfhe
#include "src/core/include/math/hal/nativeintbackend.h"  // from @openfhe
#include "src/core/include/utils/inttypes.h"             // from @openfhe

namespace mlir {
namespace heir {
namespace polynomial {

// Helper to find root of unity or return 0 if not found
llvm::APInt GetRootOfUnity(uint32_t n, const llvm::APInt& q) {
  std::optional<llvm::APInt> root = findPrimitive2nthRoot(q, n);
  if (root.has_value()) {
    return *root;
  }
  return llvm::APInt(q.getBitWidth(), 0);
}

template <uint32_t N, uint64_t Q>
static void BM_HEIR_NTT(benchmark::State& state) {
  llvm::APInt q(64, Q);
  llvm::APInt root = GetRootOfUnity(N, q);
  if (root.isZero()) {
    state.SkipWithError("Root of unity not found");
    return;
  }
  uint64_t root64 = root.getZExtValue();

  std::vector<uint64_t> coeffs(N, 0);
  for (uint32_t i = 0; i < N; ++i) {
    coeffs[i] = i % Q;
  }

  for (auto _ : state) {
    nttInPlace(coeffs, Q, root64);
    benchmark::DoNotOptimize(coeffs);
  }
}

template <uint32_t N, uint64_t Q>
static void BM_HEIR_INTT(benchmark::State& state) {
  llvm::APInt q(64, Q);
  llvm::APInt root = GetRootOfUnity(N, q);
  if (root.isZero()) {
    state.SkipWithError("Root of unity not found");
    return;
  }
  uint64_t root64 = root.getZExtValue();

  std::vector<uint64_t> coeffs(N, 0);
  for (uint32_t i = 0; i < N; ++i) {
    coeffs[i] = i % Q;
  }
  nttInPlace(coeffs, Q, root64);

  for (auto _ : state) {
    inttInPlace(coeffs, Q, root64);
    benchmark::DoNotOptimize(coeffs);
  }
}

template <uint32_t N, uint64_t Q>
static void BM_HEIR_NTT_INTT_Cycle(benchmark::State& state) {
  llvm::APInt q(64, Q);
  llvm::APInt root = GetRootOfUnity(N, q);
  if (root.isZero()) {
    state.SkipWithError("Root of unity not found");
    return;
  }
  uint64_t root64 = root.getZExtValue();

  std::vector<uint64_t> coeffs(N, 0);
  for (uint32_t i = 0; i < N; ++i) {
    coeffs[i] = i % Q;
  }

  for (auto _ : state) {
    nttInPlace(coeffs, Q, root64);
    inttInPlace(coeffs, Q, root64);
    benchmark::DoNotOptimize(coeffs);
  }
}

template <uint32_t N, uint64_t Q>
static void BM_OpenFHE_NTT_INTT_Cycle(benchmark::State& state) {
  uint32_t m = 2 * N;
  llvm::APInt qAp(64, Q);
  llvm::APInt rootAp = GetRootOfUnity(N, qAp);
  if (rootAp.isZero()) {
    state.SkipWithError("Root of unity not found");
    return;
  }

  lbcrypto::NativeInteger nativeModulus(Q);
  lbcrypto::NativeInteger nativeRoot(rootAp.getZExtValue());
  auto params =
      std::make_shared<lbcrypto::ILNativeParams>(m, nativeModulus, nativeRoot);

  lbcrypto::NativePoly openfhePoly(params, ::Format::COEFFICIENT);
  lbcrypto::NativePoly::Vector openfheVec(N, nativeModulus);
  for (uint32_t i = 0; i < N; ++i) {
    openfheVec[i] = lbcrypto::NativeInteger(i % Q);
  }
  openfhePoly.SetValues(std::move(openfheVec), ::Format::COEFFICIENT);

  for (auto _ : state) {
    openfhePoly.SwitchFormat();
    openfhePoly.SwitchFormat();
    benchmark::DoNotOptimize(openfhePoly);
  }
}

template <uint32_t N, uint64_t Q>
static void BM_OpenFHE_NTT_WithCopy(benchmark::State& state) {
  uint32_t m = 2 * N;
  llvm::APInt qAp(64, Q);
  llvm::APInt rootAp = GetRootOfUnity(N, qAp);
  if (rootAp.isZero()) {
    state.SkipWithError("Root of unity not found");
    return;
  }

  lbcrypto::NativeInteger nativeModulus(Q);
  lbcrypto::NativeInteger nativeRoot(rootAp.getZExtValue());
  auto params =
      std::make_shared<lbcrypto::ILNativeParams>(m, nativeModulus, nativeRoot);

  lbcrypto::NativePoly openfhePoly(params, ::Format::COEFFICIENT);
  lbcrypto::NativePoly::Vector openfheVec(N, nativeModulus);
  for (uint32_t i = 0; i < N; ++i) {
    openfheVec[i] = lbcrypto::NativeInteger(i % Q);
  }
  openfhePoly.SetValues(std::move(openfheVec), ::Format::COEFFICIENT);

  for (auto _ : state) {
    state.PauseTiming();
    lbcrypto::NativePoly p(openfhePoly);
    state.ResumeTiming();
    p.SwitchFormat();
    benchmark::DoNotOptimize(p);
  }
}

template <uint32_t N, uint64_t Q>
static void BM_OpenFHE_INTT_WithCopy(benchmark::State& state) {
  uint32_t m = 2 * N;
  llvm::APInt qAp(64, Q);
  llvm::APInt rootAp = GetRootOfUnity(N, qAp);
  if (rootAp.isZero()) {
    state.SkipWithError("Root of unity not found");
    return;
  }

  lbcrypto::NativeInteger nativeModulus(Q);
  lbcrypto::NativeInteger nativeRoot(rootAp.getZExtValue());
  auto params =
      std::make_shared<lbcrypto::ILNativeParams>(m, nativeModulus, nativeRoot);

  lbcrypto::NativePoly openfhePoly(params, ::Format::COEFFICIENT);
  lbcrypto::NativePoly::Vector openfheVec(N, nativeModulus);
  for (uint32_t i = 0; i < N; ++i) {
    openfheVec[i] = lbcrypto::NativeInteger(i % Q);
  }
  openfhePoly.SetValues(std::move(openfheVec), ::Format::COEFFICIENT);
  openfhePoly.SwitchFormat();

  for (auto _ : state) {
    state.PauseTiming();
    lbcrypto::NativePoly p(openfhePoly);
    state.ResumeTiming();
    p.SwitchFormat();
    benchmark::DoNotOptimize(p);
  }
}

#define REGISTER_BENCHMARKS(N, Q)                                        \
  BENCHMARK_TEMPLATE(BM_HEIR_NTT, N, Q)->Unit(benchmark::kMicrosecond);  \
  BENCHMARK_TEMPLATE(BM_HEIR_INTT, N, Q)->Unit(benchmark::kMicrosecond); \
  BENCHMARK_TEMPLATE(BM_HEIR_NTT_INTT_Cycle, N, Q)                       \
      ->Unit(benchmark::kMicrosecond);                                   \
  BENCHMARK_TEMPLATE(BM_OpenFHE_NTT_INTT_Cycle, N, Q)                    \
      ->Unit(benchmark::kMicrosecond);                                   \
  BENCHMARK_TEMPLATE(BM_OpenFHE_NTT_WithCopy, N, Q)                      \
      ->Unit(benchmark::kMicrosecond);                                   \
  BENCHMARK_TEMPLATE(BM_OpenFHE_INTT_WithCopy, N, Q)                     \
      ->Unit(benchmark::kMicrosecond);

// N = 1024
REGISTER_BENCHMARKS(1024, 65537)
REGISTER_BENCHMARKS(1024, 1152921504606846977ULL)

// N = 4096
REGISTER_BENCHMARKS(4096, 65537)
REGISTER_BENCHMARKS(4096, 1152921504606846977ULL)

// N = 8192
REGISTER_BENCHMARKS(8192, 65537)
REGISTER_BENCHMARKS(8192, 1152921504606846977ULL)

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
