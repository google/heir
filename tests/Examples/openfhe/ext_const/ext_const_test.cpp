#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/Examples/openfhe/ext_const/ext_const_lib.h"

TEST(ExtConstTest, Basic) {
  std::vector<int32_t> res = test_fn();
  std::vector<int32_t> expected = {1, 2, 3, 4};
  EXPECT_EQ(res, expected);
}
