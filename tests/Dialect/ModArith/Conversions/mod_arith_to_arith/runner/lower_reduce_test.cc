#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_reduce_1(StridedMemRefType<int32_t, 1>* result);
void _mlir_ciface_test_lower_reduce_2(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerReduceTest, Test1) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_lower_reduce_1(&result);
  ASSERT_EQ(result.sizes[0], 6);
  EXPECT_EQ(result.data[0], 3723);
  EXPECT_EQ(result.data[1], 42);
  EXPECT_EQ(result.data[2], 7679);
  EXPECT_EQ(result.data[3], 0);
  EXPECT_EQ(result.data[4], 7680);
  EXPECT_EQ(result.data[5], 7680);
  free(result.data);
}

TEST(LowerReduceTest, Test2) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_lower_reduce_2(&result);
  ASSERT_EQ(result.sizes[0], 6);
  EXPECT_EQ(result.data[0], 29498763);
  EXPECT_EQ(result.data[1], 42);
  EXPECT_EQ(result.data[2], 33554429);
  EXPECT_EQ(result.data[3], 33554430);
  EXPECT_EQ(result.data[4], 33554430);
  EXPECT_EQ(result.data[5], 7680);
  free(result.data);
}
