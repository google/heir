#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_intt_rns(StridedMemRefType<int64_t, 2>* result);
}

TEST(LowerInttRnsTest, TestInttRns) {
  StridedMemRefType<int64_t, 2> result;
  _mlir_ciface_test_intt_rns(&result);
  ASSERT_EQ(result.sizes[0], 8);
  ASSERT_EQ(result.sizes[1], 2);

  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(result.data[i * 2 + 0], i + 1)
        << "Mismatch at index " << i << " limb 0";
    EXPECT_EQ(result.data[i * 2 + 1], i + 1)
        << "Mismatch at index " << i << " limb 1";
  }

  free(result.basePtr);
}
