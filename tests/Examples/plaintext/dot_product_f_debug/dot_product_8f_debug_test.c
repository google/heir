#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

FILE *output;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} memref_1d_f32;

// This is the function we want to call from LLVM
void dot_product(
    /* result */
    memref_1d_f32 *result,
    /* arg 0*/
    float *allocated, float *aligned, int64_t offset, int64_t size,
    int64_t stride,
    /* arg 1*/
    float *allocated_2, float *aligned_2, int64_t offset_2, int64_t size_2,
    int64_t stride_2);

// Client-side encoder for each arg
void dot_product__encrypt__arg0(
    /* result */
    memref_1d_f32 *result,
    /* arg*/
    float *allocated, float *aligned, int64_t offset, int64_t size,
    int64_t stride);
void dot_product__encrypt__arg1(
    /* result */
    memref_1d_f32 *result,
    /* arg*/
    float *allocated, float *aligned, int64_t offset, int64_t size,
    int64_t stride);

// Decoders for the results
float dot_product__decrypt__result0(float *allocated, float *aligned,
                                    int64_t offset, int64_t size,
                                    int64_t stride);

// debug handler
void __heir_debug_tensor_1024xf32_(
    /* arg 0*/
    float *allocated, float *aligned, int64_t offset, int64_t size,
    int64_t stride) {
  for (int i = 0; i < size; i++) {
    fprintf(output, "%.15f ", *(aligned + i * stride));
  }
  fprintf(output, "\n");
}

void __heir_debug_f32(float value) { fprintf(output, "%.15f \n", value); }

void __heir_debug_i1(bool value) { fprintf(output, "%d \n", value); }

void __heir_debug_index(int64_t value) { fprintf(output, "%ld \n", value); }

int main(int argc, char **argv) {
  // the first argument is the output file
  if (argc > 1) {
    output = fopen(argv[1], "w");
    if (output == NULL) {
      fprintf(stderr, "Error opening file\n");
      return 1;
    }
  } else {
    output = stderr;
  }

  float arg0[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float arg1[8] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  float expected = 2.5;

  // "encrypt" each arg, which in plaintext means apply client-side encoding
  // logic
  memref_1d_f32 encArg0;
  dot_product__encrypt__arg0(&encArg0,
                             /* arg 0*/
                             arg0, arg0, 0, 8, 1);
  memref_1d_f32 encArg1;
  dot_product__encrypt__arg1(&encArg1,
                             /* arg 0*/
                             arg1, arg1, 0, 8, 1);

  // Main function
  memref_1d_f32 packedRes;
  dot_product(&packedRes,
              /* arg 0*/
              encArg0.allocated, encArg0.aligned, encArg0.offset, encArg0.size,
              encArg0.stride,
              /* arg 1*/
              encArg1.allocated, encArg1.aligned, encArg1.offset, encArg1.size,
              encArg1.stride);

  float res = dot_product__decrypt__result0(packedRes.allocated,
                                            packedRes.aligned, packedRes.offset,
                                            packedRes.size, packedRes.stride);

  if (fabs(res - expected) < 1e-3) {
    printf("Test passed\n");
#ifdef EXPECT_FAILURE
    return 1;
#endif
  } else {
    printf("Test failed %f != %f\n", res, expected);
#ifndef EXPECT_FAILURE
    return 1;
#endif
  }
}
