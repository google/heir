#include <cstdint>

#include "gtest/gtest.h" // from @googletest

extern "C" {
int32_t _mlir_ciface_lower_mod_switch_larger_width();
}

TEST(LowerModSwitchLargerWidthTest, CorrectOutput) {
  int32_t result = _mlir_ciface_lower_mod_switch_larger_width();
  EXPECT_EQ(result, 61297);
}
