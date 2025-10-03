#ifndef THIRD_PARTY_HEIR_TESTS_EXAMPLES_PLAINTEXT_DOT_PRODUCT_F_DEBUG_DEBUG_HELPER_H_
#define THIRD_PARTY_HEIR_TESTS_EXAMPLES_PLAINTEXT_DOT_PRODUCT_F_DEBUG_DEBUG_HELPER_H_

#include <cstdint>

extern "C" {
void __heir_debug_tensor_8xf32_(
    /* arg 0*/
    float* allocated, float* aligned, int64_t offset, int64_t size,
    int64_t stride);

void __heir_debug_f32(float value);

void __heir_debug_i1(bool value);

void __heir_debug_index(int64_t value);
}

#endif  // THIRD_PARTY_HEIR_TESTS_EXAMPLES_PLAINTEXT_DOT_PRODUCT_F_DEBUG_DEBUG_HELPER_H_
