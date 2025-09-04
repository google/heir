#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h" // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_mac(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMacTest, CorrectOutput) {
  StridedMemRefType<int32_t, 1> result;
  int32_t data[4];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = 4;
  result.strides[0] = 1;
  _mlir_ciface_test_lower_mac(&result);
  ASSERT_EQ(result.sizes[0], 4);
  EXPECT_EQ(result.data[0], 1600);
  EXPECT_EQ(result.data[1], 4270);
  EXPECT_EQ(result.data[2], 4);
  EXPECT_EQ(result.data[3], 3);

  free(result.data);
}
