#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_mul_ntt_roundtrip_modarith(
    StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMulNttRoundtripTest, ModArith) {
  StridedMemRefType<int32_t, 1> result{};
  _mlir_ciface_test_mul_ntt_roundtrip_modarith(&result);
  ASSERT_EQ(result.sizes[0], 8);

  free(result.basePtr);
}
