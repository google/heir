#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h" // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_leading_term(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerLeadingTermTest, TestLeadingTerm) {
  StridedMemRefType<int32_t, 1> result;
  int32_t data[2];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = 2;
  result.strides[0] = 1;
  _mlir_ciface_test_leading_term(&result);
  ASSERT_EQ(result.sizes[0], 2);
  EXPECT_EQ(result.data[0], 2);
  EXPECT_EQ(result.data[1], 10);

  free(result.data);
}
