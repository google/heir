#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_dot_product(StridedMemRefType<int16_t, 2>* result,
                              StridedMemRefType<int16_t, 2>* arg0,
                              StridedMemRefType<int16_t, 2>* arg1);

void _mlir_ciface_dot_product__encrypt__arg0(
    StridedMemRefType<int16_t, 2>* result, StridedMemRefType<int16_t>* arg);
void _mlir_ciface_dot_product__encrypt__arg1(
    StridedMemRefType<int16_t, 2>* result, StridedMemRefType<int16_t>* arg);

int16_t _mlir_ciface_dot_product__decrypt__result0(
    StridedMemRefType<int16_t, 2>* arg);
}

TEST(DotProduct8Test, Test1) {
  int16_t arg0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t arg1[8] = {2, 3, 4, 5, 6, 7, 8, 9};
  int16_t expected = 240;

  StridedMemRefType<int16_t, 2> encArg0;
  StridedMemRefType<int16_t> input0{arg0, arg0, 0, 8, 1};
  _mlir_ciface_dot_product__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<int16_t, 2> encArg1;
  StridedMemRefType<int16_t> input1{arg1, arg1, 0, 8, 1};
  _mlir_ciface_dot_product__encrypt__arg1(&encArg1, &input1);

  StridedMemRefType<int16_t, 2> packedRes;
  _mlir_ciface_dot_product(&packedRes, &encArg0, &encArg1);

  int16_t res = _mlir_ciface_dot_product__decrypt__result0(&packedRes);
  EXPECT_EQ(res, expected);
  free(encArg0.basePtr);
  free(encArg1.basePtr);
  free(packedRes.basePtr);
}
