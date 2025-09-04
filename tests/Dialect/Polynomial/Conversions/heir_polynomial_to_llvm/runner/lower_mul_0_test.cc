#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_0(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMulTest, Test0) {
  StridedMemRefType<int32_t, 1> result;
  int32_t data[12];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = 12;
  result.strides[0] = 1;
  _mlir_ciface_test_0(&result);
  ASSERT_EQ(result.sizes[0], 12);
  EXPECT_EQ(result.data[0], (int32_t)1);
  EXPECT_EQ(result.data[1], (int32_t)0);
  EXPECT_EQ(result.data[2], (int32_t)0);
  EXPECT_EQ(result.data[3], (int32_t)0);
  EXPECT_EQ(result.data[4], (int32_t)0);
  EXPECT_EQ(result.data[5], (int32_t)0);
  EXPECT_EQ(result.data[6], (int32_t)0);
  EXPECT_EQ(result.data[7], (int32_t)0);
  EXPECT_EQ(result.data[8], (int32_t)0);
  EXPECT_EQ(result.data[9], (int32_t)4294967295);
  EXPECT_EQ(result.data[10], (int32_t)1);
  EXPECT_EQ(result.data[11], (int32_t)1);

  free(result.data);
}
