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
void _mlir_ciface_bicyclic_matmul_chain(StridedMemRefType<float, 2>* res,
                                        StridedMemRefType<float, 2>* arg0,
                                        StridedMemRefType<float, 2>* arg1,
                                        StridedMemRefType<float, 2>* arg2);

void _mlir_ciface_bicyclic_matmul_chain__encrypt__arg0(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);
void _mlir_ciface_bicyclic_matmul_chain__encrypt__arg2(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);

void _mlir_ciface_bicyclic_matmul_chain__decrypt__result0(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);
}

TEST(ChainedMatmulBicyclicPlaintextTest, Test1) {
  std::vector<float> arg0(13 * 18, 0.0);
  std::vector<float> arg1(18 * 16, 0.0);
  std::vector<float> arg2(16 * 9, 0.0);

  // A[i][j] = i + j
  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 18; ++j) {
      arg0[i * 18 + j] = (i + j) / 100.0;
    }
  }

  // B[j][k] = j - k
  for (int j = 0; j < 18; ++j) {
    for (int k = 0; k < 16; ++k) {
      arg1[j * 16 + k] = (j - k) / 100.0;
    }
  }

  // V[k][l] = k - 2*l
  for (int k = 0; k < 16; ++k) {
    for (int l = 0; l < 9; ++l) {
      arg2[k * 9 + l] = (k - 2 * l) / 100.0;
    }
  }

  // Y[i][k] = sum_j A[i][j] * B[j][k]
  std::vector<float> intermediate(13 * 16, 0.0);
  for (int i = 0; i < 13; ++i) {
    for (int k = 0; k < 16; ++k) {
      for (int j = 0; j < 18; ++j) {
        intermediate[i * 16 + k] += arg0[i * 18 + j] * arg1[j * 16 + k];
      }
    }
  }

  // Z[i][l] = sum_k Y[i][k] * V[k][l]
  std::vector<float> expected(13 * 9, 0.0);
  for (int i = 0; i < 13; ++i) {
    for (int l = 0; l < 9; ++l) {
      for (int k = 0; k < 16; ++k) {
        expected[i * 9 + l] += intermediate[i * 16 + k] * arg2[k * 9 + l];
      }
    }
  }

  int64_t sizes0[2] = {13, 18};
  int64_t strides0[2] = {18, 1};
  StridedMemRefType<float, 2> inputs0(arg0.data(), arg0.data(), 0, sizes0,
                                      strides0);

  int64_t sizes1[2] = {18, 16};
  int64_t strides1[2] = {16, 1};
  StridedMemRefType<float, 2> inputs1(arg1.data(), arg1.data(), 0, sizes1,
                                      strides1);

  int64_t sizes2[2] = {16, 9};
  int64_t strides2[2] = {9, 1};
  StridedMemRefType<float, 2> inputs2(arg2.data(), arg2.data(), 0, sizes2,
                                      strides2);

  StridedMemRefType<float, 2> encArg0;
  _mlir_ciface_bicyclic_matmul_chain__encrypt__arg0(&encArg0, &inputs0);
  HEIR_MSAN_UNPOISON(&encArg0, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 2> encArg2;
  _mlir_ciface_bicyclic_matmul_chain__encrypt__arg2(&encArg2, &inputs2);
  HEIR_MSAN_UNPOISON(&encArg2, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 2> packedRes;
  _mlir_ciface_bicyclic_matmul_chain(&packedRes, &encArg0, &inputs1, &encArg2);
  HEIR_MSAN_UNPOISON(&packedRes, sizeof(StridedMemRefType<float, 2>));

  StridedMemRefType<float, 2> outRef;
  _mlir_ciface_bicyclic_matmul_chain__decrypt__result0(&outRef, &packedRes);
  HEIR_MSAN_UNPOISON(&outRef, sizeof(StridedMemRefType<float, 2>));
  HEIR_MSAN_UNPOISON(outRef.basePtr, 13 * 9 * sizeof(float));

  float errorThreshold = 1e-3;
  for (int i = 0; i < 13; ++i) {
    for (int l = 0; l < 9; ++l) {
      float actual = outRef.data[i * outRef.strides[0] + l * outRef.strides[1]];
      EXPECT_NEAR(expected[i * 9 + l], actual, errorThreshold)
          << "mismatch at (" << i << ", " << l << ")";
    }
  }

  free(encArg0.basePtr);
  free(encArg2.basePtr);
  free(packedRes.basePtr);
  free(outRef.basePtr);
}
