#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tests/Examples/plaintext/dot_product_f_debug/debug_helper.h"
#include "tests/llvm_runner/memref_types.h"

FILE* output;

extern "C" {
void _mlir_ciface_dot_product(StridedMemRefType<float>* result,
                              StridedMemRefType<float>* arg0,
                              StridedMemRefType<float>* arg1);

void _mlir_ciface_dot_product__encrypt__arg0(StridedMemRefType<float>* result,
                                             StridedMemRefType<float>* arg);

void _mlir_ciface_dot_product__encrypt__arg1(StridedMemRefType<float>* result,
                                             StridedMemRefType<float>* arg);

float _mlir_ciface_dot_product__decrypt__result0(
    StridedMemRefType<float> const*);
}

int main(int argc, char** argv) {
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

  StridedMemRefType<float> encArg0;
  StridedMemRefType<float> input0 = {arg0, arg0, 0, 8, 1};
  _mlir_ciface_dot_product__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float> encArg1;
  StridedMemRefType<float> input1 = {arg1, arg1, 0, 8, 1};
  _mlir_ciface_dot_product__encrypt__arg1(&encArg1, &input1);

  StridedMemRefType<float> packedRes;
  _mlir_ciface_dot_product(&packedRes, &encArg0, &encArg1);

  float res = _mlir_ciface_dot_product__decrypt__result0(&packedRes);

  free(encArg0.basePtr);
  free(encArg1.basePtr);
  free(packedRes.basePtr);

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
