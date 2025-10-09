#include "tests/Examples/plaintext/dot_product_f_debug/debug_helper.h"

#include <cstdint>
#include <cstdio>

#include "tests/llvm_runner/memref_types.h"

extern FILE* output;

extern "C" {
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
