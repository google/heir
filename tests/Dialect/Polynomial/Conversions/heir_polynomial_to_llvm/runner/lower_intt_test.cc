#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_intt(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerInttTest, TestIntt) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_intt(&result);
  ASSERT_EQ(result.sizes[0], 4);
  EXPECT_EQ(result.data[0], 1);
  EXPECT_EQ(result.data[1], 2);
  EXPECT_EQ(result.data[2], 3);
  EXPECT_EQ(result.data[3], 4);
  free(result.basePtr);
}
