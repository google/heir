#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_0(StridedMemRefType<int64_t, 1>* result);
}

TEST(LowerMulTest, Test0) {
  StridedMemRefType<int64_t, 1> result;
  _mlir_ciface_test_0(&result);
  ASSERT_EQ(result.sizes[0], 12);
  EXPECT_EQ(result.data[0], (int64_t)1);
  EXPECT_EQ(result.data[1], (int64_t)0);
  EXPECT_EQ(result.data[2], (int64_t)0);
  EXPECT_EQ(result.data[3], (int64_t)0);
  EXPECT_EQ(result.data[4], (int64_t)0);
  EXPECT_EQ(result.data[5], (int64_t)0);
  EXPECT_EQ(result.data[6], (int64_t)0);
  EXPECT_EQ(result.data[7], (int64_t)0);
  EXPECT_EQ(result.data[8], (int64_t)0);
  EXPECT_EQ(result.data[9], (int64_t)4294967295);
  EXPECT_EQ(result.data[10], (int64_t)1);
  EXPECT_EQ(result.data[11], (int64_t)1);
  free(result.basePtr);
}
