#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/TransformUtils.h"

namespace mlir {
namespace heir {

namespace {

TEST(TransformUtilsTest, GetMinUnusedInputEmptyPermutation) {
  std::vector<int64_t> perm;
  EXPECT_EQ(getMinUnusedInput(perm), -1);
}

TEST(TransformUtilsTest, GetMinUnusedInputAllUnset) {
  std::vector<int64_t> perm = {-1, -1, -1};
  EXPECT_EQ(getMinUnusedInput(perm), 0);
}

TEST(TransformUtilsTest, GetMinUnusedInputReturnsFirstUnsetIndex) {
  std::vector<int64_t> perm = {2, 0, -1, 1, -1};
  EXPECT_EQ(getMinUnusedInput(perm), 2);
}

TEST(TransformUtilsTest, GetMinUnusedInputLastIndexUnset) {
  std::vector<int64_t> perm = {1, 2, 0, -1};
  EXPECT_EQ(getMinUnusedInput(perm), 3);
}

TEST(TransformUtilsTest, GetMinUnusedInputFullPermutation) {
  std::vector<int64_t> perm = {2, 0, 1};
  EXPECT_EQ(getMinUnusedInput(perm), -1);
}

TEST(TransformUtilsTest, GetMinUnusedTargetReturnsSmallestUnmappedOutput) {
  std::vector<int64_t> perm = {2, 0, -1, -1};
  EXPECT_EQ(getMinUnusedTarget(perm), 1);
}

TEST(TransformUtilsTest, GetMinUnusedTargetFullPermutation) {
  std::vector<int64_t> perm = {2, 0, 1};
  EXPECT_EQ(getMinUnusedTarget(perm), -1);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
