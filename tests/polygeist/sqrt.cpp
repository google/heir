/*
* Compile this file to mlir using
*

   Polygeist/build/bin/cgeist \
    '-function=*' \
    -raise-scf-to-affine \
    --memref-fullrank \
    -S \
    -O3 \
    sqrt.cc

*/

int isqrt(int num) {
  int res = 0;
  int bit = 1 << 14;  // ((unsigned) INT16_MAX + 1) / 2.

  for (int i = 0; i < 8; ++i) {
    if (num >= res + bit) {
      num -= res + bit;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  return res;
}
