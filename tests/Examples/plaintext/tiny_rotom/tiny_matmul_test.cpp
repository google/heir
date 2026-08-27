#include <cstdint>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define HEIR_MSAN_UNPOISON(p, s) __msan_unpoison((p), (s))
#else
#define HEIR_MSAN_UNPOISON(p, s)
#endif
#else
#define HEIR_MSAN_UNPOISON(p, s)
#endif

extern "C" {
void _mlir_ciface_tiny(StridedMemRefType<float, 2>* res,
                       StridedMemRefType<float, 2>* w,
                       StridedMemRefType<float, 2>* b,
                       StridedMemRefType<float, 2>* x);
void _mlir_ciface_tiny__packed_plaintext__arg0(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);
void _mlir_ciface_tiny__packed_plaintext__arg1(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 1>* arg);
void _mlir_ciface_tiny__encrypt__arg2(StridedMemRefType<float, 2>* res,
                                      StridedMemRefType<float, 2>* arg);
void _mlir_ciface_tiny__decrypt__result0(StridedMemRefType<float, 2>* res,
                                         StridedMemRefType<float, 2>* arg);
}

namespace {
constexpr int64_t kIn = 16, kOut = 8;

StridedMemRefType<float, 2> memref2(std::vector<float>& d, int64_t r,
                                    int64_t c) {
  int64_t sizes[2] = {r, c};
  int64_t strides[2] = {c, 1};
  return StridedMemRefType<float, 2>(d.data(), d.data(), 0, sizes, strides);
}
StridedMemRefType<float, 1> memref1(std::vector<float>& d, int64_t n) {
  int64_t sizes[1] = {n};
  int64_t strides[1] = {1};
  return StridedMemRefType<float, 1>(d.data(), d.data(), 0, sizes, strides);
}
}  // namespace

TEST(TinyRotomPlaintextTest, MatchesReference) {
  std::vector<float> w(kOut * kIn), b(kOut), x(kIn);
  for (int64_t j = 0; j < kOut; ++j) {
    b[j] = 0.01f * static_cast<float>(j + 1);
    for (int64_t k = 0; k < kIn; ++k)
      w[j * kIn + k] = 0.01f * ((j + k) % 7 - 3);
  }
  for (int64_t k = 0; k < kIn; ++k) x[k] = 0.1f * ((k % 5) + 1);

  std::vector<float> expected(kOut);
  for (int64_t j = 0; j < kOut; ++j) {
    float acc = b[j];
    for (int64_t k = 0; k < kIn; ++k) acc += w[j * kIn + k] * x[k];
    expected[j] = acc;
  }

  auto wRef = memref2(w, kOut, kIn);
  auto bRef = memref1(b, kOut);
  auto xRef = memref2(x, 1, kIn);
  StridedMemRefType<float, 2> wP, bP, xE, resP, out;
  _mlir_ciface_tiny__packed_plaintext__arg0(&wP, &wRef);
  HEIR_MSAN_UNPOISON(&wP, sizeof(wP));
  _mlir_ciface_tiny__packed_plaintext__arg1(&bP, &bRef);
  HEIR_MSAN_UNPOISON(&bP, sizeof(bP));
  _mlir_ciface_tiny__encrypt__arg2(&xE, &xRef);
  HEIR_MSAN_UNPOISON(&xE, sizeof(xE));
  _mlir_ciface_tiny(&resP, &wP, &bP, &xE);
  HEIR_MSAN_UNPOISON(&resP, sizeof(resP));
  _mlir_ciface_tiny__decrypt__result0(&out, &resP);
  HEIR_MSAN_UNPOISON(&out, sizeof(out));
  HEIR_MSAN_UNPOISON(out.basePtr, kOut * sizeof(float));

  for (int64_t j = 0; j < kOut; ++j) {
    float actual = out.data[0 * out.strides[0] + j * out.strides[1]];
    EXPECT_NEAR(expected[j], actual, 1e-3) << "mismatch at output " << j;
  }
  free(wP.basePtr);
  free(xE.basePtr);
  free(resP.basePtr);
  free(out.basePtr);
}
