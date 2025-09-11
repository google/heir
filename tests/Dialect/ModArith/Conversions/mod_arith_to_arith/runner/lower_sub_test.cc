#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_sub(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerSubTest, CorrectOutput) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_lower_sub(&result);
  ASSERT_EQ(result.sizes[0], 4);
  EXPECT_EQ(result.data[0], 6188);
  EXPECT_EQ(result.data[1], 489);
  EXPECT_EQ(result.data[2], 7680);
  EXPECT_EQ(result.data[3], 0);
  free(result.data);
}
