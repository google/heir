#include <stdint.h>
#include <stdio.h>

typedef struct Memref1DI16 {
  int16_t* allocated;
  int16_t* aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} memref_i16_1d;

typedef struct Memref1DI64 {
  int64_t* allocated;
  int64_t* aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} memref_i64_1d;

// This is the function we want to call from LLVM
// For mod-arith, the width is 64 bits
/* result <nxi64> */
memref_i64_1d roberts_cross(
    /* arg0 <nxi64> */
    int64_t* allocated, int64_t* aligned, int64_t offset, int64_t size,
    int64_t stride);

// Client-side encoder for each arg
// Note that our input is i16 but encoder makes it i64 due to mod-arith
/* result <nxi64> */
memref_i64_1d roberts_cross__encrypt__arg0(
    /* arg <nxi16> */
    int16_t* allocated, int16_t* aligned, int64_t offset, int64_t size,
    int64_t stride);

// Decoders for the results
// Note that our output is i16 but decoder takes i64 due to mod-arith
/* result <nxi16> */
memref_i16_1d roberts_cross__decrypt__result0(
    /* arg <nxi64> */
    int64_t* allocated, int64_t* aligned, int64_t offset, int64_t size,
    int64_t stride);

void memrefCopy();

int main() {
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

  // "encrypt" each arg, which in plaintext means apply client-side encoding
  // logic
  memref_i64_1d encArg0 = roberts_cross__encrypt__arg0(input, input, 0, 256, 1);

  memref_i64_1d memref = roberts_cross(
      /* arg 0*/
      encArg0.allocated, encArg0.aligned, encArg0.offset, encArg0.size,
      encArg0.stride);

  memref_i16_1d decRes = roberts_cross__decrypt__result0(
      memref.allocated, memref.aligned, memref.offset, memref.size,
      memref.stride);

  int16_t* res = decRes.aligned;

  for (int i = 0; i != 256; ++i) {
    if (res[i] != expected[i]) {
      printf("Test failed at %d: %hd != %hd\n", i, res[i], expected[i]);
#ifdef EXPECT_FAILURE
      return 0;
#else
      return 1;
#endif
    }
  }
  printf("Test passed\n");
#ifdef EXPECT_FAILURE
  return 1;
#else
  return 0;
#endif
}
