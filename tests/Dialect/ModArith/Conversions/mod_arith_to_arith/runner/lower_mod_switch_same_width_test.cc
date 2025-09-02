#include "gtest/gtest.h"

extern "C" {
int32_t _mlir_ciface_lower_mod_switch_same_width();
}

TEST(LowerModSwitchSameWidthTest, CorrectOutput) {
  int32_t result = _mlir_ciface_lower_mod_switch_same_width();
  EXPECT_EQ(result, 32325566);
}
