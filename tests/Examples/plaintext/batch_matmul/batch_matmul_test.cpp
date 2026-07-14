#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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
void _mlir_ciface_batch_matmul(StridedMemRefType<float, 2>* res,
                               StridedMemRefType<float, 2>* arg0,
                               StridedMemRefType<float, 2>* arg1);

void _mlir_ciface_batch_matmul__encrypt__arg0(StridedMemRefType<float, 2>* res,
                                              StridedMemRefType<float, 3>* arg);
void _mlir_ciface_batch_matmul__encrypt__arg1(StridedMemRefType<float, 2>* res,
                                              StridedMemRefType<float, 3>* arg);

void _mlir_ciface_batch_matmul__decrypt__result0(
    StridedMemRefType<float, 3>* res, StridedMemRefType<float, 2>* arg);
}

TEST(BatchMatmulPlaintextRobustTest, Test1) {
  std::vector<float> arg0(2 * 17 * 19, 0.0);
  std::vector<float> arg1(2 * 19 * 21, 0.0);

  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 17; ++i) {
      for (int j = 0; j < 19; ++j) {
        arg0[b * 17 * 19 + i * 19 + j] = (b + i + j) / 100.0;
      }
    }
  }

  for (int b = 0; b < 2; ++b) {
    for (int j = 0; j < 19; ++j) {
      for (int k = 0; k < 21; ++k) {
        arg1[b * 19 * 21 + j * 21 + k] = (b + j - k) / 100.0;
      }
    }
  }

  std::vector<float> expected(2 * 17 * 21, 0.0);
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 17; ++i) {
      for (int k = 0; k < 21; ++k) {
        for (int j = 0; j < 19; ++j) {
          expected[b * 17 * 21 + i * 21 + k] +=
              arg0[b * 17 * 19 + i * 19 + j] * arg1[b * 19 * 21 + j * 21 + k];
        }
      }
    }
  }

  int64_t sizes0[3] = {2, 17, 19};
  int64_t strides0[3] = {17 * 19, 19, 1};
  StridedMemRefType<float, 3> inputs0(arg0.data(), arg0.data(), 0, sizes0,
                                      strides0);

  int64_t sizes1[3] = {2, 19, 21};
  int64_t strides1[3] = {19 * 21, 21, 1};
  StridedMemRefType<float, 3> inputs1(arg1.data(), arg1.data(), 0, sizes1,
                                      strides1);

  StridedMemRefType<float, 2> encArg0;
  _mlir_ciface_batch_matmul__encrypt__arg0(&encArg0, &inputs0);
  HEIR_MSAN_UNPOISON(&encArg0, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 2> encArg1;
  _mlir_ciface_batch_matmul__encrypt__arg1(&encArg1, &inputs1);
  HEIR_MSAN_UNPOISON(&encArg1, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 2> packedRes;
  _mlir_ciface_batch_matmul(&packedRes, &encArg0, &encArg1);
  HEIR_MSAN_UNPOISON(&packedRes, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 3> outRef;
  _mlir_ciface_batch_matmul__decrypt__result0(&outRef, &packedRes);
  HEIR_MSAN_UNPOISON(&outRef, sizeof(StridedMemRefType<float, 3>));
  HEIR_MSAN_UNPOISON(outRef.basePtr, 2 * 17 * 21 * sizeof(float));

  float errorThreshold = 1e-3;
  for (int i = 0; i < 2 * 17 * 21; ++i) {
    EXPECT_NEAR(expected[i], outRef.basePtr[i], errorThreshold);
  }

  free(encArg0.basePtr);
  free(encArg1.basePtr);
  free(packedRes.basePtr);
  free(outRef.basePtr);
}
