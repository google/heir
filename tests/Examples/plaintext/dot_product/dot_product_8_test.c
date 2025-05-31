#include <stdint.h>
#include <stdio.h>

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} memref_1d;

// This is the function we want to call from LLVM
void dot_product(memref_1d *result,
                 /* arg 0*/
                 int16_t *allocated, int16_t *aligned, int64_t offset,
                 int64_t size, int64_t stride,
                 /* arg 1*/
                 int16_t *allocated_2, int16_t *aligned_2, int64_t offset_2,
                 int64_t size_2, int64_t stride_2);

// Client-side encoder for each arg
void dot_product__encrypt__arg0(
    /* result */
    memref_1d *result,
    /* arg*/
    int16_t *allocated, int16_t *aligned, int64_t offset, int64_t size,
    int64_t stride);
void dot_product__encrypt__arg1(
    /* result */
    memref_1d *result,
    /* arg*/
    int16_t *allocated, int16_t *aligned, int64_t offset, int64_t size,
    int64_t stride);

// Decoders for the results
int16_t dot_product__decrypt__result0(int16_t *allocated, int16_t *aligned,
                                      int64_t offset, int64_t size,
                                      int64_t stride);

int main() {
  int16_t arg0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t arg1[8] = {2, 3, 4, 5, 6, 7, 8, 9};
  int16_t expected = 240;

  // "encrypt" each arg, which in plaintext means apply client-side encoding
  // logic
  memref_1d encArg0;
  dot_product__encrypt__arg0(&encArg0,
                             /* arg 0*/
                             arg0, arg0, 0, 8, 1);
  memref_1d encArg1;
  dot_product__encrypt__arg1(&encArg1,
                             /* arg 0*/
                             arg1, arg1, 0, 8, 1);

  memref_1d packedRes;
  dot_product(&packedRes,
              /* arg 0*/
              encArg0.allocated, encArg0.aligned, encArg0.offset, encArg0.size,
              encArg0.stride,
              /* arg 1*/
              encArg1.allocated, encArg1.aligned, encArg1.offset, encArg1.size,
              encArg1.stride);

  int16_t res = dot_product__decrypt__result0(
      packedRes.allocated, packedRes.aligned, packedRes.offset, packedRes.size,
      packedRes.stride);

  if (res == expected) {
    printf("Test passed\n");
#ifdef EXPECT_FAILURE
    return 1;
#endif
  } else {
    printf("Test failed %d != %d\n", res, expected);
#ifndef EXPECT_FAILURE
    return 1;
#endif
  }
}
