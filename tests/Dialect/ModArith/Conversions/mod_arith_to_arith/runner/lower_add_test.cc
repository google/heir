#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_add(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerAddTest, CorrectOutput) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_lower_add(&result);
  ASSERT_EQ(result.sizes[0], 4);
  EXPECT_EQ(result.data[0], 1258);
  EXPECT_EQ(result.data[1], 7276);
  EXPECT_EQ(result.data[2], 7678);
  EXPECT_EQ(result.data[3], 0);
  free(result.data);
}
