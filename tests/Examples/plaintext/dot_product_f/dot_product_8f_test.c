#include <math.h>
#include <stdint.h>
#include <stdio.h>

// This is the function we want to call from LLVM
float dot_product(
    /* arg 0*/
    float *allocated, float *aligned, int64_t offset, int64_t size,
    int64_t stride,
    /* arg 1*/
    float *allocated_2, float *aligned_2, int64_t offset_2, int64_t size_2,
    int64_t stride_2);

int main() {
  float arg0[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float arg1[8] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  float expected = 2.5;

  float res = dot_product(
      /* arg 0*/
      arg0, arg0, 0, 8, 1,
      /* arg 1*/
      arg1, arg1, 0, 8, 1);

  if (fabs(res - expected) < 1e-3) {
    printf("Test passed\n");
  } else {
    printf("Test failed %f != %f\n", res, expected);
  }
}
