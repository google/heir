/*
 * This file is a debug helper for IntegerRelations that represent ciphertext
 * packings. Given an IntegerRelation, you can call
 * Codegen.h::generateLoopNestAsCStr in this directory to generate a C string
 * that can be copied here and run to simulate the generated packing.
 *
 * Note that if the dimensions of the test relation differ, this file will need
 * to be modified to match.
 *
 * This is the 1D version with a 1D data array.
 */
#include <stdio.h>

#define SIZE 8
#define CTS 1
#define SLOTS 16

int data[SIZE];
int ciphertexts[CTS][SLOTS];

void S(int d0, int d1, int d2) { ciphertexts[d1][d2] = data[d0]; }

void print1D(int* array, int size) {
  for (int i = 0; i < size; i++) {
    printf("%4d ", array[i]);
  }
  printf("\n");
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
  // Data array is 0, 1, 2, ...
  for (int i = 0; i < SIZE; i++) {
    data[i] = i;
  }
  printf("Data array:\n");
  print1D(data, SIZE);

  for (int i = 0; i < CTS; i++) {
    for (int j = 0; j < SLOTS; j++) {
      ciphertexts[i][j] = 0;
    }
  }

  // Insert the codegenned loop here from debug output.
  // clang-format off
  for (int c1 = 0; c1 <= 15; c1 += 1)
    S((c1 + 3) % 8, 0, c1);
  // clang-format ont

  printf("Packed ciphertexts:\n");
  print2D((int*)ciphertexts, CTS, SLOTS);

  return 0;
}
