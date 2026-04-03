#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_mul_ntt_roundtrip_modarith(
    StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMulNttRoundtripTest, ModArith) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_mul_ntt_roundtrip_modarith(&result);
  ASSERT_EQ(result.sizes[0], 8);
  EXPECT_EQ(result.data[0], 1);
  EXPECT_EQ(result.data[1], 2);
  EXPECT_EQ(result.data[2], 3);
  EXPECT_EQ(result.data[3], 8);
  EXPECT_EQ(result.data[4], 7);
  EXPECT_EQ(result.data[5], 6);
  EXPECT_EQ(result.data[6], 3);
  EXPECT_EQ(result.data[7], 4);
  free(result.basePtr);
}
