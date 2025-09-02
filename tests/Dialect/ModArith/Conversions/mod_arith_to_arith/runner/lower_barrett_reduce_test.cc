#include "gtest/gtest.h"
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_barrett_reduce(
    StridedMemRefType<int32_t, 1> *result);
}

TEST(LowerBarrettReduceTest, CorrectOutput) {
  StridedMemRefType<int32_t, 1> result;
  int32_t data[4];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = 4;
  result.strides[0] = 1;
  _mlir_ciface_test_lower_barrett_reduce(&result);
  ASSERT_EQ(result.sizes[0], 4);
  EXPECT_EQ(result.data[0], 3723);
  EXPECT_EQ(result.data[1], 7680);
  EXPECT_EQ(result.data[2], 17);
  EXPECT_EQ(result.data[3], 7681);
}
