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
void _mlir_ciface_bicyclic_matmul_pt(StridedMemRefType<float, 2>* res,
                                     StridedMemRefType<float, 2>* arg0,
                                     StridedMemRefType<float, 2>* arg1);

void _mlir_ciface_bicyclic_matmul_pt__encrypt__arg0(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);

void _mlir_ciface_bicyclic_matmul_pt__decrypt__result0(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);
}

TEST(BicyclicMatmulPtPlaintextRobustTest, Test1) {
  std::vector<float> arg0(13 * 14, 0.0);
  std::vector<float> arg1(14 * 16, 0.0);

  // A[i][j] = i + j
  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 14; ++j) {
      arg0[i * 14 + j] = (i + j) / 100.0;
    }
  }

  // B[j][k] = j - k
  for (int j = 0; j < 14; ++j) {
    for (int k = 0; k < 16; ++k) {
      arg1[j * 16 + k] = (j - k) / 100.0;
    }
  }

  // Precompute expected C[i][k] from actual arg0 and arg1 vectors
  std::vector<float> expected(13 * 16, 0.0);
  for (int i = 0; i < 13; ++i) {
    for (int k = 0; k < 16; ++k) {
      for (int j = 0; j < 14; ++j) {
        expected[i * 16 + k] += arg0[i * 14 + j] * arg1[j * 16 + k];
      }
    }
  }

  int64_t sizes0[2] = {13, 14};
  int64_t strides0[2] = {14, 1};
  StridedMemRefType<float, 2> inputs0(arg0.data(), arg0.data(), 0, sizes0,
                                      strides0);

  int64_t sizes1[2] = {14, 16};
  int64_t strides1[2] = {16, 1};
  StridedMemRefType<float, 2> inputs1(arg1.data(), arg1.data(), 0, sizes1,
                                      strides1);

  StridedMemRefType<float, 2> encArg0;
  _mlir_ciface_bicyclic_matmul_pt__encrypt__arg0(&encArg0, &inputs0);
  HEIR_MSAN_UNPOISON(&encArg0, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 2> packedRes;
  // Note: arg1 is passed as a plaintext memref directly
  _mlir_ciface_bicyclic_matmul_pt(&packedRes, &encArg0, &inputs1);
  HEIR_MSAN_UNPOISON(&packedRes, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 2> outRef;
  _mlir_ciface_bicyclic_matmul_pt__decrypt__result0(&outRef, &packedRes);
  HEIR_MSAN_UNPOISON(&outRef, sizeof(StridedMemRefType<float, 2>));
  HEIR_MSAN_UNPOISON(outRef.basePtr, 13 * 16 * sizeof(float));

  float errorThreshold = 1e-3;
  for (int i = 0; i < 13 * 16; ++i) {
    if (std::abs(outRef.basePtr[i] - expected[i]) > errorThreshold) {
      std::cerr << "Decryption error at index " << i << ": "
                << outRef.basePtr[i] << " != " << expected[i] << std::endl;
      FAIL();
    }
  }

  free(encArg0.basePtr);
  free(packedRes.basePtr);
  free(outRef.basePtr);
}
