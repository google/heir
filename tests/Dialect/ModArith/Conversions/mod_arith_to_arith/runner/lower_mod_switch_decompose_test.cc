#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_test_lower_mod_switch_decompose(
    StridedMemRefType<int16_t, 1>* result);
}

TEST(LowerModSwitchDecomposeTest, CorrectOutput) {
  StridedMemRefType<int16_t, 1> result;
  _mlir_ciface_test_lower_mod_switch_decompose(&result);
  ASSERT_EQ(result.sizes[0], 3);
  EXPECT_EQ(result.data[0], 265);
  EXPECT_EQ(result.data[1], 43);
  EXPECT_EQ(result.data[2], 7);
  free(result.data);
}
