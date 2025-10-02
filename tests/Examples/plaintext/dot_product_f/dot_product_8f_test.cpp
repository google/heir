#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_dot_product(StridedMemRefType<float>* res,
                              StridedMemRefType<float>* arg0,
                              StridedMemRefType<float>* arg1);

void _mlir_ciface_dot_product__encrypt__arg0(StridedMemRefType<float>* res,
                                             StridedMemRefType<float>* arg);
void _mlir_ciface_dot_product__encrypt__arg1(StridedMemRefType<float>* res,
                                             StridedMemRefType<float>* arg);

float _mlir_ciface_dot_product__decrypt__result0(StridedMemRefType<float>* arg);
}

TEST(DotProductFTest, Test1) {
  float arg0[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float arg1[8] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  float expected = 2.5;

  StridedMemRefType<float> encArg0;
  StridedMemRefType<float> inputs0 = {arg0, arg0, 0, 8, 1};
  _mlir_ciface_dot_product__encrypt__arg0(&encArg0, &inputs0);

  StridedMemRefType<float> encArg1;
  StridedMemRefType<float> inputs1 = {arg1, arg1, 0, 8, 1};
  _mlir_ciface_dot_product__encrypt__arg0(&encArg1, &inputs1);

  StridedMemRefType<float> packedRes;
  _mlir_ciface_dot_product(&packedRes, &encArg0, &encArg1);

  float res = _mlir_ciface_dot_product__decrypt__result0(&packedRes);

  EXPECT_TRUE(std::fabs(res - expected) < 1e-3);
  free(encArg0.basePtr);
  free(encArg1.basePtr);
  free(packedRes.basePtr);
}
