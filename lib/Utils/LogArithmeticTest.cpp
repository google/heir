#include <cmath>
#include <limits>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/LogArithmetic.h"

namespace mlir {
namespace heir {
namespace {

TEST(ArithmeticDagTest, TestStorage) {
  auto zero = Log2Arithmetic::of(0.0);
  // Check the storage value
  EXPECT_EQ(zero.getLog2Value(), -std::numeric_limits<double>::infinity());
  EXPECT_EQ(zero.getValue(), 0.0);

  auto one = Log2Arithmetic::of(1.0);
  // Check the storage value
  EXPECT_EQ(one.getLog2Value(), 0.0);
  EXPECT_EQ(one.getValue(), 1.0);
}

TEST(ArithmeticDagTest, TestZeroArith) {
  auto zero = Log2Arithmetic::of(0.0);
  EXPECT_EQ(zero + zero, zero);
  EXPECT_EQ(zero * zero, zero);
}

TEST(ArithmeticDagTest, TestOneArith) {
  auto zero = Log2Arithmetic::of(0.0);
  auto one = Log2Arithmetic::of(1.0);
  auto two = Log2Arithmetic::of(2.0);
  EXPECT_EQ(zero + one, one);
  EXPECT_EQ(zero * one, zero);
  EXPECT_EQ(one + one, two);
  EXPECT_EQ(one * one, one);
}

TEST(ArithmeticDagTest, TestLargeArith) {
  auto l512 = Log2Arithmetic::of(std::pow(2, 512));
  // Check the storage value
  EXPECT_EQ((l512 * l512 * l512 * l512).getLog2Value(), 2048.0);
  EXPECT_EQ((l512 + l512 + l512 + l512).getLog2Value(), 514.0);
}

TEST(ArithmeticDagTest, TestNearZeroArith) {
  auto l512 = Log2Arithmetic::of(std::pow(2, -512));
  // Check the storage value
  EXPECT_EQ((l512 * l512 * l512 * l512).getLog2Value(), -2048.0);
  EXPECT_EQ((l512 + l512 + l512 + l512).getLog2Value(), -510.0);
}

TEST(ArithmeticDagTest, TestZeroCompareOne) {
  auto zero = Log2Arithmetic::of(0.0);
  auto one = Log2Arithmetic::of(1.0);
  EXPECT_LT(zero, one);
  EXPECT_FALSE(one < zero);
}

TEST(ArithmeticDagTest, TestZeroCompareZero) {
  auto zero = Log2Arithmetic::of(0.0);
  EXPECT_FALSE(zero < zero);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
