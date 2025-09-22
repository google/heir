/*
 * This file is a debug helper for IntegerRelations that represent ciphertext
 * packings. Given an IntegerRelation, you can call
 * Codegen.h::generateLoopNestAsCStr in this directory to generate a C string
 * that can be copied here and run to simulate the generated packing.
 *
 * Note that if the dimensions of the test relation differ, this file will need
 * to be modified to match.
 */
#include <stdio.h>

#include <algorithm>

#define ROWS 3
#define COLS 3
#define CTS 9
#define SLOTS 9

int data[ROWS][COLS];
int ciphertexts[CTS][SLOTS];

void S(int d0, int d1, int d2, int d3) {
  printf("(%4d, %4d) -> (%4d, %4d)\n", d0, d1, d2, d3);
  ciphertexts[d2][d3] = data[d0][d1];
}

void print2D(int* matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%4d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {
  // Data matrix is 0, 1, 2, ...
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      data[i][j] = i * COLS + j + 1;
    }
  }

  for (int i = 0; i < CTS; i++) {
    for (int j = 0; j < SLOTS; j++) {
      ciphertexts[i][j] = 0;
    }
  }

  // Insert the codegenned loop here from debug output.
  // clang-format off
  for (int c0 = 0; c0 <= 8; c0 += 1)
    for (int c1 = std::max(std::max(std::max(-1, c0 - 4), -(c0 % 3) + c0 - 3), -(c0 % 3)); c1 <= std::min(std::min(std::min(9, c0 + 4), -(c0 % 3) + c0 + 5), -(c0 % 3) + 10); c1 += 1)
        if (c1 + 1 >= (-c0 + c1 + 4) % 3 && ((-c0 + c1 + 4) % 3) + 7 >= c1 && ((-c0 + c1 + 4) % 3) + (c0 % 3) >= 1 && ((-c0 + c1 + 4) % 3) + (c0 % 3) <= 3)
              S((-c0 + c1 + 4) / 3, (-c0 + c1 + 4) % 3, c0, c1);

  // clang-format on

  printf("Data matrix:\n");
  print2D((int*)data, ROWS, COLS);

  printf("Packed ciphertexts:\n");
  print2D((int*)ciphertexts, CTS, SLOTS);

  return 0;
}
