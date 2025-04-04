#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Transforms/LayoutPropagation/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using llvm::SmallVector;

TEST(UtilsTest, TestShiftByInserted1) {
  SmallVector<int64_t> dims = {0, 1, 2, 3};
  SmallVector<int64_t> inserts = {1, 2};
  SmallVector<int64_t> expected = {0, 3, 4, 5};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByInserted2) {
  SmallVector<int64_t> dims = {2, 6, 7, 8};
  SmallVector<int64_t> inserts = {0, 4};
  SmallVector<int64_t> expected = {3, 8, 9, 10};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByInsertedCollision) {
  SmallVector<int64_t> dims = {3, 6, 7, 8};
  SmallVector<int64_t> inserts = {0, 4};
  SmallVector<int64_t> expected = {5, 8, 9, 10};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemoved1) {
  SmallVector<int64_t> dims = {0, 3, 4, 5};
  SmallVector<int64_t> removals = {1, 2};
  SmallVector<int64_t> expected = {0, 1, 2, 3};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemoved2) {
  SmallVector<int64_t> dims = {3, 8, 9, 10};
  SmallVector<int64_t> removals = {0, 4};
  SmallVector<int64_t> expected = {2, 6, 7, 8};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemovedCollision) {
  SmallVector<int64_t> dims = {5, 8, 9, 10};
  SmallVector<int64_t> removals = {0, 4};
  SmallVector<int64_t> expected = {3, 6, 7, 8};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
