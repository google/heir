#include <cstdint>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Conversion/MemrefToArith/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {

using ::llvm::SmallVector;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(FlattenIndex, UnflattenOffsetZero) {
  SmallVector<int64_t, 3> strides = {20, 5, 1};
  static const int64_t offset = 0;
  EXPECT_THAT(unflattenIndex(33, strides, offset), ElementsAre(1, 2, 3));
}

TEST(FlattenIndex, OffsetZero) {
  static const llvm::SmallVector<int64_t> strides = {20, 5, 1};
  static const llvm::SmallVector<int64_t> indices = {1, 2, 3};
  static const int64_t offset = 0;
  EXPECT_EQ(33, flattenIndex(indices, strides, offset));
}

TEST(FlattenIndex, UnflattenOffsetNonzero) {
  static const llvm::SmallVector<int64_t> strides = {20, 5, 1};
  static const llvm::SmallVector<int64_t> expected = {1, 2, 1};
  static const int64_t offset = 2;
  llvm::SmallVector<int64_t> actual = unflattenIndex(33, strides, offset);
  EXPECT_THAT(actual, ElementsAre(1, 2, 1));
}

TEST(FlattenIndex, OffsetNonzero) {
  static const llvm::SmallVector<int64_t> strides = {20, 5, 1};
  static const llvm::SmallVector<int64_t> indices = {1, 2, 1};
  static const int64_t offset = 2;
  EXPECT_EQ(33, flattenIndex(indices, strides, offset));
}

TEST(FlattenIndex, FlattenUnflattenSame) {
  static const llvm::SmallVector<int64_t> strides = {10, 1};
  static const llvm::SmallVector<int64_t> indices = {7, 0};
  static const int64_t offset = 13;
  auto flattened = flattenIndex(indices, strides, offset);
  EXPECT_EQ(83, flattened);
  auto unflattened = unflattenIndex(flattened, strides, offset);
  EXPECT_THAT(unflattened, ElementsAreArray(indices));
}

TEST(FlattenIndex, FlattenUnflattenIntoFlat) {
  static const llvm::SmallVector<int64_t> startingStrides = {10, 1};
  static const llvm::SmallVector<int64_t> startingIndices = {7, 0};
  static const int64_t startingOffset = 13;
  auto flattened =
      flattenIndex(startingIndices, startingStrides, startingOffset);

  static const llvm::SmallVector<int64_t> endingStrides = {1};
  static const int64_t endingOffset = 0;
  auto unflattened = unflattenIndex(flattened, endingStrides, endingOffset);

  EXPECT_THAT(unflattened, ElementsAre(83));
}

}  // namespace heir
}  // namespace mlir
