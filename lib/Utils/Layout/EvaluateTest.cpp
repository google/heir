#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;
using ::testing::Eq;

TEST(EvaluateTest, EvaluateLayoutOnVectorTrivial) {
  IntegerRelation relation =
      getIntegerRelationFromIslStr(
          "{ [d0] -> [ct, d1] : d0 - d1 = 0 and d0 >= 0 and d1 >= 0 and 10 >= "
          "d0 and 10 >= d1 and ct = 0 }")
          .value();
  std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto result = evaluateLayoutOnVector(relation, input);
  ASSERT_THAT(result.size(), Eq(1));
  ASSERT_THAT(result[0], Eq(input));
}

TEST(EvaluateTest, EvaluateLayoutOnVectorRowMajor) {
  // Data of size 32 being packed into ciphertexts of size 1024.
  auto relation =
      getIntegerRelationFromIslStr(
          "{ [row] -> [ct, slot] : (slot - row) mod 32 = 0 and row >= 0 and ct "
          ">= 0 and slot >= 0 and 1023 >= slot and 0 >= ct and 31 >= row }")
          .value();

  std::vector<int> input;
  input.reserve(32);
  for (int i = 0; i < 32; i++) {
    input.push_back(i);
  }

  auto result = evaluateLayoutOnVector(relation, input);
  ASSERT_THAT(result.size(), Eq(1));
  ASSERT_THAT(result[0].size(), Eq(1024));
  for (int i = 0; i < 32; i++) {
    std::vector<int> slice = {result[0].begin() + i * 32,
                              result[0].begin() + i * 32 + 32};
    ASSERT_THAT(slice, Eq(input));
  }
}

TEST(EvaluateTest, EvaluateLayoutFor2DConv) {
  // Figure 3 in Orion: https://arxiv.org/pdf/2311.03470
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  auto relation =
      get2dConvFilterRelation(filterType, dataType, strides, /*padding=*/1);

  std::vector<std::vector<int>> filter = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  // The expanded matrix will have size 9x9,
  std::vector<std::vector<int>> expected = {
      {5, 6, 0, 8, 9, 0, 0, 0, 0}, {4, 5, 6, 7, 8, 9, 0, 0, 0},
      {0, 4, 5, 0, 7, 8, 0, 0, 0}, {2, 3, 0, 5, 6, 0, 8, 9, 0},
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 0, 4, 5, 0, 7, 8},
      {0, 0, 0, 2, 3, 0, 5, 6, 0}, {0, 0, 0, 1, 2, 3, 4, 5, 6},
      {0, 0, 0, 0, 1, 2, 0, 4, 5}};

  auto result = evaluateLayoutOnMatrix(relation, filter);
  ASSERT_THAT(result, Eq(expected));
}

TEST(EvaluateTest, EvaluateLayoutFor2DConv3x3Data2x2Filter) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  IntegerRelation relation =
      get2dConvFilterRelation(filterType, dataType, strides, /*padding=*/1);

  std::vector<std::vector<int>> filter = {{1, 2}, {3, 4}};
  std::vector<std::vector<int>> expected = {
      {4, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 4, 0, 0, 0, 0, 0, 0, 0},
      {0, 3, 4, 0, 0, 0, 0, 0, 0}, {0, 0, 3, 0, 0, 0, 0, 0, 0},
      {2, 0, 0, 4, 0, 0, 0, 0, 0}, {1, 2, 0, 3, 4, 0, 0, 0, 0},
      {0, 1, 2, 0, 3, 4, 0, 0, 0}, {0, 0, 1, 0, 0, 3, 0, 0, 0},
      {0, 0, 0, 2, 0, 0, 4, 0, 0}, {0, 0, 0, 1, 2, 0, 3, 4, 0},
      {0, 0, 0, 0, 1, 2, 0, 3, 4}, {0, 0, 0, 0, 0, 1, 0, 0, 3},
      {0, 0, 0, 0, 0, 0, 2, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 0},
      {0, 0, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 0, 0, 1},
  };

  auto result = evaluateLayoutOnMatrix(relation, filter);
  ASSERT_THAT(result, Eq(expected));
}

TEST(EvaluateTest, EvaluateLayoutFor2DConv3x4Data2x2Filter) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 4}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  IntegerRelation relation =
      get2dConvFilterRelation(filterType, dataType, strides, /*padding=*/1);

  std::vector<std::vector<int>> filter = {{1, 2}, {3, 4}};
  std::vector<std::vector<int>> expected = {
      {4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0},

      {2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0},
      {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0},
      {0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0},
      {0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0},
      {0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0},

      {0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0},
      {0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0},
      {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0},
      {0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4},
      {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3},

      {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
  };

  auto result = evaluateLayoutOnMatrix(relation, filter);
  ASSERT_THAT(result, Eq(expected));
}

TEST(EvaluateTest, EvaluateLayoutFor2DConvDiagonalized) {
  // Figure 3 in Orion: https://arxiv.org/pdf/2311.03470
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  auto relation = getConvFilterDiagonalizedRelation(
      filterType, dataType, /*padding=*/1, /*ciphertextSize=*/16);

  std::vector<std::vector<int>> filter = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  // The expanded matrix will have size 9x9, and diagonalizing it will require
  // padding with zeros.
  std::vector<std::vector<int>> expected = {
      {5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0},
      {6, 6, 0, 6, 6, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 7, 7, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {9, 9, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0}};

  auto result = evaluateLayoutOnMatrix(relation.value(), filter);
  ASSERT_THAT(result, Eq(expected));
}

TEST(EvaluateTest, EvaluateLayoutFor2DConvChwFchw) {
  MLIRContext context;
  // Orion example in Figure 4 of https://arxiv.org/pdf/2311.03470.
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  int64_t padding = 1;
  IntegerRelation rel =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);

  std::vector<std::vector<std::vector<std::vector<int>>>> filter = {
      {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}},
      {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}};

  std::function<int(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };

  auto result = evaluateLayout(rel, getValueFn);

  std::vector<std::vector<int>> expected = {
      {5, 6, 0, 8, 9, 0, 0, 0, 0, 5, 6, 0, 8, 9, 0, 0, 0, 0},
      {4, 5, 6, 7, 8, 9, 0, 0, 0, 4, 5, 6, 7, 8, 9, 0, 0, 0},
      {0, 4, 5, 0, 7, 8, 0, 0, 0, 0, 4, 5, 0, 7, 8, 0, 0, 0},
      {2, 3, 0, 5, 6, 0, 8, 9, 0, 2, 3, 0, 5, 6, 0, 8, 9, 0},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 1, 2, 0, 4, 5, 0, 7, 8},
      {0, 0, 0, 2, 3, 0, 5, 6, 0, 0, 0, 0, 2, 3, 0, 5, 6, 0},
      {0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 1, 2, 3, 4, 5, 6},
      {0, 0, 0, 0, 1, 2, 0, 4, 5, 0, 0, 0, 0, 1, 2, 0, 4, 5},
      {5, 6, 0, 8, 9, 0, 0, 0, 0, 5, 6, 0, 8, 9, 0, 0, 0, 0},
      {4, 5, 6, 7, 8, 9, 0, 0, 0, 4, 5, 6, 7, 8, 9, 0, 0, 0},
      {0, 4, 5, 0, 7, 8, 0, 0, 0, 0, 4, 5, 0, 7, 8, 0, 0, 0},
      {2, 3, 0, 5, 6, 0, 8, 9, 0, 2, 3, 0, 5, 6, 0, 8, 9, 0},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 1, 2, 0, 4, 5, 0, 7, 8},
      {0, 0, 0, 2, 3, 0, 5, 6, 0, 0, 0, 0, 2, 3, 0, 5, 6, 0},
      {0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 1, 2, 3, 4, 5, 6},
      {0, 0, 0, 0, 1, 2, 0, 4, 5, 0, 0, 0, 0, 1, 2, 0, 4, 5}};

  ASSERT_THAT(result, Eq(expected));
}

TEST(EvaluateTest, EvaluateLayoutFor2DConvChwFchwNoPadding) {
  MLIRContext context;
  // Filter 2x2 and data is 4x4 so there are 2x2 sliding windows.
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 4, 4}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  IntegerRelation rel =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);

  std::vector<std::vector<std::vector<std::vector<int>>>> filter = {
      {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
      {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}};
  std::function<int(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };

  auto result = evaluateLayout(rel, getValueFn);

  std::vector<std::vector<int>> expected = {
      {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8},
      {9,  10, 0, 0, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       13, 14, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 9,  10, 0, 0, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 13, 14, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 9,  10, 0, 0, 11, 12, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 0, 0, 15, 16, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,  10, 0, 0, 11, 12,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 0, 0, 15, 16},
  };

  ASSERT_THAT(result, Eq(expected));
}

TEST(EvaluateTest, EvaluateLayoutFor2DConvChwFchwNoPaddingDiagonalized) {
  MLIRContext context;
  // Filter 2x2 and data is 4x4 so there are 2x2 sliding windows.
  RankedTensorType filterType =
      RankedTensorType::get({4, 1, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 1, 4, 4}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  auto rel = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, 16, false);
  ASSERT_TRUE(succeeded(rel));

  std::vector<std::vector<std::vector<std::vector<int>>>> filter = {
      {{{1, 2}, {3, 4}}},  // Channel 0
      {{{1, 2}, {3, 4}}},  // Channel 1
      {{{1, 2}, {3, 4}}},  // Channel 2
      {{{1, 2}, {3, 4}}}   // Channel 3
  };
  std::function<int(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };

  auto result = evaluateLayout(rel.value(), getValueFn);

  std::vector<std::vector<int>> expected = {
      {1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4},
      {2, 1, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 2, 0, 0, 0, 4, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 4, 3, 0, 0, 0, 0},
      {3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 1, 0, 0, 0},
      {4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0},
      {0, 4, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0},
      {0, 0, 2, 1, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 2, 0, 0, 0, 4, 1, 0, 0, 0, 3, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 4, 3, 0, 0},
      {0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 1, 0},
      {0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1},
      {0, 0, 0, 4, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2},
      {0, 0, 0, 0, 2, 1, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 1, 0, 0, 0, 3, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 4, 3}};
  EXPECT_THAT(result, Eq(expected));

  // Now test minimal non zero diagonals
  auto relOptimized = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, 16, true);
  ASSERT_TRUE(succeeded(relOptimized));
  auto resultOptimized = evaluateLayout(relOptimized.value(), getValueFn);

  std::vector<std::vector<int>> expectedOptimized = {
      {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4},
      {2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 4, 0, 4, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0},
      {3, 4, 3, 4, 0, 0, 0, 0, 3, 4, 3, 4, 0, 0, 0, 0},
      {4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1},
      {0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 1, 2, 1, 2},
      {0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 1, 0, 1, 0, 3, 0, 3, 0, 1, 0, 1, 0, 3, 0, 3}};
  EXPECT_THAT(resultOptimized, Eq(expectedOptimized));
}

TEST(EvaluateTest, Conv2dResultRelation) {
  MLIRContext context;
  RankedTensorType outputType =
      RankedTensorType::get({1, 4, 2, 2}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;

  // Fits in one ciphertext
  int64_t ciphertextSize = 16;
  IntegerRelation rel =
      get2dConvResultRelation(outputType, strides, padding, ciphertextSize);
  EXPECT_EQ(rel.getNumDomainVars(), outputType.getRank());
  EXPECT_EQ(rel.getNumRangeVars(), 2);

  std::vector<std::vector<std::vector<std::vector<int>>>> output = {
      {{{1, 2}, {3, 4}},
       {{5, 6}, {7, 8}},
       {{9, 10}, {11, 12}},
       {{13, 14}, {15, 16}}}};
  std::function<int(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return output[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };
  auto result = evaluateLayout(rel, getValueFn);
  std::vector<std::vector<int>> expected = {
      {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}};
  EXPECT_THAT(result, Eq(expected));
}

TEST(EvaluateTest, Conv2dResultRelationTwoCiphertexts) {
  MLIRContext context;
  RankedTensorType outputType =
      RankedTensorType::get({1, 4, 2, 2}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;

  int64_t ciphertextSize = 8;
  IntegerRelation rel =
      get2dConvResultRelation(outputType, strides, padding, ciphertextSize);

  std::vector<std::vector<std::vector<std::vector<int>>>> output = {
      {{{1, 2}, {3, 4}},
       {{5, 6}, {7, 8}},
       {{9, 10}, {11, 12}},
       {{13, 14}, {15, 16}}}};
  std::function<int(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return output[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };
  auto result = evaluateLayout(rel, getValueFn);
  std::vector<std::vector<int>> expected = {{1, 5, 2, 6, 9, 13, 10, 14},
                                            {3, 7, 4, 8, 11, 15, 12, 16}};
  EXPECT_THAT(result, Eq(expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
