#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_5(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMulTest, Test5) {
  StridedMemRefType<int32_t, 1> result;
  int32_t data[12];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = 12;
  result.strides[0] = 1;
  _mlir_ciface_test_5(&result);
  ASSERT_EQ(result.sizes[0], 12);
  EXPECT_EQ(result.data[0], (int32_t)1);
  EXPECT_EQ(result.data[1], (int32_t)0);
  EXPECT_EQ(result.data[2], (int32_t)1);
  EXPECT_EQ(result.data[3], (int32_t)1);
  EXPECT_EQ(result.data[4], (int32_t)0);
  EXPECT_EQ(result.data[5], (int32_t)1);

  free(result.data);
}
