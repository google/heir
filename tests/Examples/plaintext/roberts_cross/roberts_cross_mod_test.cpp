#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
// For mod-arith, the width is 64 bits
void _mlir_ciface_roberts_cross(StridedMemRefType<int64_t>* result,
                                StridedMemRefType<int64_t>* arg0);

// Note that our input is i16 but encoder makes it i64 due to mod-arith
void _mlir_ciface_roberts_cross__encrypt__arg0(
    StridedMemRefType<int64_t>* result, StridedMemRefType<int16_t>* arg);

// Note that our output is i16 but decoder takes i64 due to mod-arith
void _mlir_ciface_roberts_cross__decrypt__result0(
    StridedMemRefType<int16_t>* result, StridedMemRefType<int64_t>* arg);
}

TEST(RobertsCrossModTest, Test1) {
  int16_t input[256];
  int16_t expected[256];

  for (int i = 0; i < 256; ++i) {
    input[i] = i;
  }

  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      // (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
      int64_t xY = (row * 16 + col) % 256;
      int64_t xYm1 = (row * 16 + col - 1) % 256;
      int64_t xm1Y = ((row - 1) * 16 + col) % 256;
      int64_t xm1Ym1 = ((row - 1) * 16 + col - 1) % 256;

      if (xYm1 < 0) xYm1 += 256;
      if (xm1Y < 0) xm1Y += 256;
      if (xm1Ym1 < 0) xm1Ym1 += 256;

      int16_t v1 = (input[xm1Ym1] - input[xY]);
      int16_t v2 = (input[xm1Y] - input[xYm1]);
      int16_t sum = v1 * v1 + v2 * v2;
      expected[row * 16 + col] = sum;
    }
  }

  StridedMemRefType<int64_t> encArg0;
  StridedMemRefType<int16_t> input0{input, input, 0, 256, 1};
  _mlir_ciface_roberts_cross__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<int64_t> memref;
  _mlir_ciface_roberts_cross(&memref, &encArg0);

  StridedMemRefType<int16_t> decRes;
  _mlir_ciface_roberts_cross__decrypt__result0(&decRes, &memref);

  int16_t* res = decRes.data;

#ifdef EXPECT_FAILURE
  EXPECT_FALSE(std::equal(res, res + 256, expected));
#else
  EXPECT_TRUE(std::equal(res, res + 256, expected));
#endif
  free(encArg0.basePtr);
  free(memref.basePtr);
  free(decRes.basePtr);
}
