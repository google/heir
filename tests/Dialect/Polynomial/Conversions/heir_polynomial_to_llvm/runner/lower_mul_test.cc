#include "gtest/gtest.h"
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_mul(StridedMemRefType<int32_t, 1> *result);
}

TEST(LowerMulTest, TestMul) {
  StridedMemRefType<int32_t, 1> result;
  int32_t data[12];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = 12;
  result.strides[0] = 1;
  _mlir_ciface_test_mul(&result);
  ASSERT_EQ(result.sizes[0], 12);
  EXPECT_EQ(result.data[0], 1);
  EXPECT_EQ(result.data[1], 0);
  EXPECT_EQ(result.data[2], 0);
  EXPECT_EQ(result.data[3], 0);
  EXPECT_EQ(result.data[4], 0);
  EXPECT_EQ(result.data[5], 0);
  EXPECT_EQ(result.data[6], 0);
  EXPECT_EQ(result.data[7], 0);
  EXPECT_EQ(result.data[8], 0);
  EXPECT_EQ(result.data[9], 65535);
  EXPECT_EQ(result.data[10], 1);
  EXPECT_EQ(result.data[11], 1);
}
