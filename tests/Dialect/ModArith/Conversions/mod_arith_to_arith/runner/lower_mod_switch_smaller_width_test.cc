#include "gtest/gtest.h"

extern "C" {
int16_t _mlir_ciface_lower_mod_switch_smaller_width();
}

TEST(LowerModSwitchSmallerWidthTest, CorrectOutput) {
  int16_t result = _mlir_ciface_lower_mod_switch_smaller_width();
  EXPECT_EQ(result & 0x3ff, 197);
}
