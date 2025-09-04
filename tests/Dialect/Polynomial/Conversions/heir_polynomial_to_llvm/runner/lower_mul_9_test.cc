#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h" // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_9(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMulTest, Test9) {
  StridedMemRefType<int32_t, 1> result;
  int32_t data[3];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = 3;
  result.strides[0] = 1;
  _mlir_ciface_test_9(&result);
  ASSERT_EQ(result.sizes[0], 3);
  EXPECT_EQ(result.data[0], (int32_t)4);
  EXPECT_EQ(result.data[1], (int32_t)3);
  EXPECT_EQ(result.data[2], (int32_t)3);

  free(result.data);
}
