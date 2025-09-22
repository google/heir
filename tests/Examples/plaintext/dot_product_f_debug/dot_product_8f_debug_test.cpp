#include <cmath>
#include <cstdint>
#include <cstdio>

#include "gtest/gtest.h"  // from @googletest
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

// debug handler
void __heir_debug_tensor_8xf32_(
    /* arg 0*/
    float* allocated, float* aligned, int64_t offset, int64_t size,
    int64_t stride) {
  for (int i = 0; i < size; i++) {
    std::fprintf(output, "%.15f ", *(aligned + i * stride));
  }
  std::fprintf(output, "\n");
}

void __heir_debug_f32(float value) { std::fprintf(output, "%.15f \n", value); }

void __heir_debug_i1(bool value) { std::fprintf(output, "%d \n", value); }

void __heir_debug_index(int64_t value) {
  std::fprintf(output, "%ld \n", value);
}
}

int g_argc;
char** g_argv;

TEST(DotProduct8FDebugTest, Test1) {
  // the first argument is the output file
  if (g_argc > 1) {
    output = fopen(g_argv[1], "w");
    if (output == NULL) {
      GTEST_FAIL() << "Error opening file";
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
  _mlir_ciface_dot_product__encrypt__arg0(&encArg1, &input1);

  StridedMemRefType<float> packedRes;
  _mlir_ciface_dot_product(&packedRes, &encArg0, &encArg1);

  float res = _mlir_ciface_dot_product__decrypt__result0(&packedRes);

  EXPECT_TRUE(std::fabs(res - expected) < 1e-3);
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
