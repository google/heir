#include <stdint.h>
#include <stdio.h>

// This is the function we want to call from LLVM
// For mod-arith, the width is 64 bits
int64_t dot_product(
    /* arg 0*/
    int64_t *allocated, int64_t *aligned, int64_t offset, int64_t size,
    int64_t stride,
    /* arg 1*/
    int64_t *allocated_2, int64_t *aligned_2, int64_t offset_2, int64_t size_2,
    int64_t stride_2);

int main() {
  int64_t arg0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t arg1[8] = {2, 3, 4, 5, 6, 7, 8, 9};
  int64_t expected = 240;

  int64_t res = dot_product(
      /* arg 0*/
      arg0, arg0, 0, 8, 1,
      /* arg 1*/
      arg1, arg1, 0, 8, 1);

  if (res == expected) {
    printf("Test passed\n");
  } else {
    printf("Test failed %ld != %ld\n", res, expected);
  }
}
