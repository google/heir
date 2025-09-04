#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_8(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMulTest, Test8) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_8(&result);
  ASSERT_EQ(result.sizes[0], 3);
  EXPECT_EQ(result.data[0], (int32_t)4);
  EXPECT_EQ(result.data[1], (int32_t)1);
  EXPECT_EQ(result.data[2], (int32_t)3);
  free(result.basePtr);
}
