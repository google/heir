#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_ntt(StridedMemRefType<int32_t, 1>* result);
}

TEST(LowerNttTest, TestNtt) {
  StridedMemRefType<int32_t, 1> result;
  _mlir_ciface_test_ntt(&result);
  ASSERT_EQ(result.sizes[0], 4);
  EXPECT_EQ(result.data[0], 1467);
  EXPECT_EQ(result.data[1], 2807);
  EXPECT_EQ(result.data[2], 3471);
  EXPECT_EQ(result.data[3], 7621);
  free(result.basePtr);
}
