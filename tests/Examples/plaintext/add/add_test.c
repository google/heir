#include <stdint.h>
#include <stdio.h>

struct Memref1D {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

// This is the function we want to call from LLVM
struct Memref1D add(
    /* arg 0*/
    int16_t *allocated, int16_t *aligned, int64_t offset, int64_t size,
    int64_t stride);
//     /* arg 1*/
//     int16_t *allocated_2, int16_t *aligned_2, int64_t offset_2, int64_t
//     size_2, int64_t stride_2);

int main() {
  int16_t arg0[1024] = {
      1, 2, 3, 4, 5, 6, 7, 8,
  };
  int16_t expected = 1;

  struct Memref1D res = add(
      /* arg 0*/
      arg0, arg0, 0, 1024, 1);
  //       /* arg 1*/
  //       arg1, arg1, 0, 8, 1);

  int64_t result = *(res.aligned);

  if (result == expected) {
    printf("Test passed\n");
#ifdef EXPECT_FAILURE
    return 1;
#endif
  } else {
    printf("Test failed %d != %d\n", result, expected);
#ifndef EXPECT_FAILURE
    return 1;
#endif
  }
}
