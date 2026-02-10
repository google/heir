#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "tests/llvm_runner/memref_types.h"

FILE* output;

extern "C" {
void _mlir_ciface_loop(StridedMemRefType<float, 2>* result,
                       StridedMemRefType<float, 2>* arg0);

void _mlir_ciface_loop__encrypt__arg0(StridedMemRefType<float, 2>* result,
                                      StridedMemRefType<float>* arg);

void _mlir_ciface_loop__decrypt__result0(StridedMemRefType<float, 2>* result,
                                         StridedMemRefType<float, 2>* arg);

void __heir_debug_tensor_8xf32_(
    /* arg 0*/
    float* allocated, float* aligned, int64_t offset, int64_t size,
    int64_t stride) {
  for (int i = 0; i < size; i++) {
    std::fprintf(output, "%.10f ", *(aligned + i * stride));
  }
  std::fprintf(output, "\n");
}

// debug handler
void __heir_debug_tensor_1x8xf32_(
    /* arg 0*/
    float* allocated, float* aligned, int64_t offset, int64_t size,
    int64_t stride) {
  for (int i = 0; i < size; i++) {
    __heir_debug_tensor_8xf32_(allocated, aligned + i * stride, offset, 8, 1);
  }
}

void __heir_debug_f32(float value) { std::fprintf(output, "%.15f \n", value); }

void __heir_debug_i1(bool value) { std::fprintf(output, "%d \n", value); }

void __heir_debug_index(int64_t value) {
  std::fprintf(output, "%ld \n", value);
}
}

int main(int argc, char** argv) {
  // the first argument is the output file
  if (argc > 1) {
    output = fopen(argv[1], "w");
    if (output == nullptr) {
      fprintf(stderr, "Error opening file\n");
      return 1;
    }
  } else {
    output = stderr;
  }

  float arg0[8] = {0.,         0.14285714, 0.28571429, 0.42857143,
                   0.57142857, 0.71428571, 0.85714286, 1.};
  float expected[8] = {-1.,         -1.16666629, -1.39989342, -1.74687019,
                       -2.29543899, -3.19507837, -4.66914279, -7.};

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0{arg0, arg0, 0, 8, 1};
  _mlir_ciface_loop__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float, 2> packedRes;
  _mlir_ciface_loop(&packedRes, &encArg0);

  StridedMemRefType<float, 2> decRes;
  _mlir_ciface_loop__decrypt__result0(&decRes, &packedRes);
  float* res = decRes.data;

  for (int i = 0; i < 8; ++i) {
    std::cout << "res[" << i << "] = " << res[i] << ", expected[" << i
              << "] = " << expected[i] << std::endl;

    if (fabs(res[i] - expected[i]) > 1e-5) {
      printf("Test failed %f != %f\n", res[i], expected[i]);
      return 1;
    }
  }

  free(encArg0.basePtr);
  free(packedRes.basePtr);
  free(decRes.basePtr);

  return 0;
}
