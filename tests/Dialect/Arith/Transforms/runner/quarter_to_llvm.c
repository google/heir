#include <stdint.h>
#include <stdio.h>

// This is the function we want to call from LLVM
void test_lowera_quarter_mul(uint16_t *res, uint16_t i0, uint16_t i1,
                             uint16_t i2, uint16_t i3);

int main(int argc, char *argv[]) {
  uint16_t c[4] = {0};

  test_lowera_quarter_mul(c, 251, 0, 0, 0);

  printf("%x %x %x %x", c[3], c[2], c[1], c[0]);

  return 0;
}
