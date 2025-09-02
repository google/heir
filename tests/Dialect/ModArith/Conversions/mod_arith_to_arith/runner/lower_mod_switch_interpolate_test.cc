#include "gtest/gtest.h"

extern "C" {
int32_t _mlir_ciface_test_lower_mod_switch_interpolate();
}

TEST(LowerModSwitchInterpolateTest, CorrectOutput) {
  int32_t result = _mlir_ciface_test_lower_mod_switch_interpolate();
  EXPECT_EQ(result, 1113316);
}
