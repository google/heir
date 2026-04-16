#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_ntt_rns(StridedMemRefType<int64_t, 2>* result);
}

TEST(LowerNttRnsTest, TestNttRns) {
  StridedMemRefType<int64_t, 2> result;
  _mlir_ciface_test_ntt_rns(&result);
  ASSERT_EQ(result.sizes[0], 8);
  ASSERT_EQ(result.sizes[1], 2);

  int64_t expected_q17[] = {5, 9, 13, 5, 0, 11, 8, 8};
  int64_t expected_q97[] = {86, 4, 41, 53, 56, 4, 67, 85};

  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(result.data[i * 2 + 0], expected_q17[i])
        << "Mismatch at index " << i << " limb 0";
    EXPECT_EQ(result.data[i * 2 + 1], expected_q97[i])
        << "Mismatch at index " << i << " limb 1";
  }

  free(result.basePtr);
}
