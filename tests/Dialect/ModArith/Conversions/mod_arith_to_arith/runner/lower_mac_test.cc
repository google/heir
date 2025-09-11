#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_mac(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerMacTest, CorrectOutput) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_lower_mac(&result);
  ASSERT_EQ(result.sizes[0], 4);
  EXPECT_EQ(result.data[0], 1600);
  EXPECT_EQ(result.data[1], 4270);
  EXPECT_EQ(result.data[2], 4);
  EXPECT_EQ(result.data[3], 3);
  free(result.data);
}
