#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_lift_centered(
    StridedMemRefType<int32_t, 1>* result);
void _mlir_ciface_test_lower_lift_standard(
    StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerLiftCenteredTest, CorrectOutput) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_lower_lift_centered(&result);
  ASSERT_EQ(result.sizes[0], 14);
  EXPECT_EQ(result.data[0], 0);
  EXPECT_EQ(result.data[1], 1);
  EXPECT_EQ(result.data[2], -1);
  EXPECT_EQ(result.data[3], 0);
  EXPECT_EQ(result.data[4], 1);
  EXPECT_EQ(result.data[5], -1);
  EXPECT_EQ(result.data[6], 0);
  EXPECT_EQ(result.data[7], 1);
  EXPECT_EQ(result.data[8], -2);
  EXPECT_EQ(result.data[9], -1);
  EXPECT_EQ(result.data[10], 0);
  EXPECT_EQ(result.data[11], 1);
  EXPECT_EQ(result.data[12], -2);
  EXPECT_EQ(result.data[13], -1);
  free(result.basePtr);
}

TEST(LowerLiftStandardTest, CorrectOutput) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_lower_lift_standard(&result);
  ASSERT_EQ(result.sizes[0], 14);
  EXPECT_EQ(result.data[0], 0);
  EXPECT_EQ(result.data[1], 1);
  EXPECT_EQ(result.data[2], 2);
  EXPECT_EQ(result.data[3], 0);
  EXPECT_EQ(result.data[4], 1);
  EXPECT_EQ(result.data[5], 2);
  EXPECT_EQ(result.data[6], 0);
  EXPECT_EQ(result.data[7], 1);
  EXPECT_EQ(result.data[8], 2);
  EXPECT_EQ(result.data[9], 3);
  EXPECT_EQ(result.data[10], 0);
  EXPECT_EQ(result.data[11], 1);
  EXPECT_EQ(result.data[12], 2);
  EXPECT_EQ(result.data[13], 3);
  free(result.basePtr);
}
