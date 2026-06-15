#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_bicyclic_matmul(StridedMemRefType<float, 2>* res,
                                  StridedMemRefType<float, 2>* arg0,
                                  StridedMemRefType<float, 2>* arg1);

void _mlir_ciface_bicyclic_matmul__encrypt__arg0(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);
void _mlir_ciface_bicyclic_matmul__encrypt__arg1(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);

void _mlir_ciface_bicyclic_matmul__decrypt__result0(
    StridedMemRefType<float, 2>* res, StridedMemRefType<float, 2>* arg);
}

TEST(BicyclicMatmulPlaintextRobustTest, Test1) {
  std::vector<float> arg0(16 * 17, 0.0);
  std::vector<float> arg1(17 * 19, 0.0);

  // A[i][j] = i + j
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 17; ++j) {
      arg0[i * 17 + j] = i + j;
    }
  }

  // B[j][k] = j - k
  for (int j = 0; j < 17; ++j) {
    for (int k = 0; k < 19; ++k) {
      arg1[j * 19 + k] = j - k;
    }
  }

  // Precompute expected C[i][k] from actual arg0 and arg1 vectors
  std::vector<float> expected(16 * 19, 0.0);
  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 19; ++k) {
      for (int j = 0; j < 17; ++j) {
        expected[i * 19 + k] += arg0[i * 17 + j] * arg1[j * 19 + k];
      }
    }
  }

  int64_t sizes0[2] = {16, 17};
  int64_t strides0[2] = {17, 1};
  StridedMemRefType<float, 2> inputs0(arg0.data(), arg0.data(), 0, sizes0,
                                      strides0);

  int64_t sizes1[2] = {17, 19};
  int64_t strides1[2] = {19, 1};
  StridedMemRefType<float, 2> inputs1(arg1.data(), arg1.data(), 0, sizes1,
                                      strides1);

  StridedMemRefType<float, 2> encArg0;
  _mlir_ciface_bicyclic_matmul__encrypt__arg0(&encArg0, &inputs0);

  StridedMemRefType<float, 2> encArg1;
  _mlir_ciface_bicyclic_matmul__encrypt__arg1(&encArg1, &inputs1);

  StridedMemRefType<float, 2> packedRes;
  _mlir_ciface_bicyclic_matmul(&packedRes, &encArg0, &encArg1);

  StridedMemRefType<float, 2> outRef;
  _mlir_ciface_bicyclic_matmul__decrypt__result0(&outRef, &packedRes);

  float errorThreshold = 1e-3;
  for (int i = 0; i < 16 * 19; ++i) {
    if (std::abs(outRef.basePtr[i] - expected[i]) > errorThreshold) {
      std::cerr << "Decryption error at index " << i << ": "
                << outRef.basePtr[i] << " != " << expected[i] << std::endl;
      FAIL();
    }
  }

  free(encArg0.basePtr);
  free(encArg1.basePtr);
  free(packedRes.basePtr);
  free(outRef.basePtr);
}
