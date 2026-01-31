#include <cstdlib>
#include <iostream>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_loop(StridedMemRefType<float, 2>* result,
                       StridedMemRefType<float, 2>* arg0);

void _mlir_ciface_loop__encrypt__arg0(StridedMemRefType<float, 2>* result,
                                      StridedMemRefType<float>* arg);

void _mlir_ciface_loop__decrypt__result0(StridedMemRefType<float, 2>* result,
                                         StridedMemRefType<float, 2>* arg);
}

TEST(LoopTest, Test1) {
  float arg0[8] = {0.,         0.14285714, 0.28571429, 0.42857143,
                   0.57142857, 0.71428571, 0.85714286, 1.};
  float expected[8] = {-1.,         -1.16666629, -1.39989342, -1.74687019,
                       -2.29543899, -3.19507837, -4.66914279, -7.};

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0{arg0, arg0, 0, 8, 1};
  _mlir_ciface_loop__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float, 2> packedRes;
  _mlir_ciface_loop(&packedRes, &encArg0);

  StridedMemRefType<float, 2> decRes;
  _mlir_ciface_loop__decrypt__result0(&decRes, &packedRes);
  float* res = decRes.data;

  for (int i = 0; i < 8; ++i) {
    std::cout << "res[" << i << "] = " << res[i] << ", expected[" << i
              << "] = " << expected[i] << std::endl;
    EXPECT_NEAR(res[i], expected[i], 1e-5);
  }

  free(encArg0.basePtr);
  free(packedRes.basePtr);
  free(decRes.basePtr);
}
