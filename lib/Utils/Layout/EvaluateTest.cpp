#include <string>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;
using ::testing::Eq;

TEST(CodegenTest, EvaluateLayoutOnVectorTrivial) {
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

TEST(CodegenTest, EvaluateLayoutOnVectorRowMajor) {
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

TEST(CodegenTest, EvaluateLayoutFor2DConv) {
  // Figure 3 in Orion: https://arxiv.org/pdf/2311.03470
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  auto relation = get2dConvFilterRelation(filterType, dataType, /*padding=*/1);

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

TEST(CodegenTest, EvaluateLayoutFor2DConvDiagonalized) {
  // Figure 3 in Orion: https://arxiv.org/pdf/2311.03470
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  auto relation = get2dConvFilterDiagonalizedRelation(
      filterType, dataType, /*padding=*/1, /*ciphertextSize=*/16);

  std::vector<std::vector<int>> filter = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  // The expanded matrix will have size 9x9, and diagonalizing it will require
  // padding with zeros.
  std::vector<std::vector<int>> expected = {
      {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
      {6, 6, 0, 6, 6, 0, 6, 0, 6, 6, 0, 6, 6, 0, 6, 0},
      {0, 7, 7, 0, 7, 7, 0, 0, 0, 7, 7, 0, 7, 7, 0, 0},
      {8, 8, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0},
      {9, 9, 0, 9, 1, 1, 0, 1, 9, 9, 0, 9, 1, 1, 0, 1},
      {0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2},
      {0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 3, 3, 0, 3, 3},
      {0, 4, 4, 0, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4}};

  auto result = evaluateLayoutOnMatrix(relation.value(), filter);
  ASSERT_THAT(result, Eq(expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
