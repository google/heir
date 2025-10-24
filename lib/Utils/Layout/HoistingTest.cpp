#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

// ISL

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;

TEST(HoistingTest, DiagonalLayoutRotateVecSlots) {
  MLIRContext context;
  // In this test, the vector is laid out row-major, and the conversion rotates
  // the slots by 3. This should hoist to a rotation of each diagonally-laid-out
  // vector by 3.
  auto maybeFromVecLayout = getIntegerRelationFromIslStr(
      "{ [d] ->  [ct, slot] : (d - slot) mod 8 = 0 and d >= 0 and 7 >= d and "
      "slot >= 0 and 15 >= slot and ct = 0 }");
  ASSERT_TRUE(succeeded(maybeFromVecLayout));
  auto fromVecLayout = maybeFromVecLayout.value();
  auto maybeToVecLayout = getIntegerRelationFromIslStr(
      "{ [d] ->  [ct, slot] : (d - (slot + 3)) mod 8 = 0 and d >= 0 and 7 >= d "
      "and slot >= 0 and 15 >= slot and ct = 0 }");
  ASSERT_TRUE(succeeded(maybeToVecLayout));
  auto toVecLayout = maybeToVecLayout.value();
  auto maybeMatrixLayout = getIntegerRelationFromIslStr(
      "{ [row, col] -> [ct, slot] : (slot - row) mod 8 = 0 and (ct + slot - "
      "col) mod 8 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and "
      "15 >= slot and 7 >= ct and 7 >= row and 7 >= col }");
  ASSERT_TRUE(succeeded(maybeMatrixLayout));
  auto matrixLayout = maybeMatrixLayout.value();
  const IntegerRelation expected =
      getIntegerRelationFromIslStr(
          "{ [row, col] -> [ct, slot] : ((slot + 3) - row) mod 8 = 0 and "
          "(ct + (slot + 3) - col) mod 8 = 0 and row >= 0 and col >= 0 and ct "
          ">= 0 and slot >= 0 and 15 >= slot and 7 >= ct and 7 >= row and 7 >= "
          "col }")
          .value();

  IntegerRelation actual =
      hoistConversionThroughMatvec(matrixLayout, fromVecLayout, toVecLayout);

  // Spot check some points
  std::vector<std::pair<int, int>> expectedDomain = {
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
  };
  std::vector<std::pair<int, int>> expectedRange = {
      {0, 5}, {0, 6}, {0, 7}, {0, 0}, {7, 1}, {0, 1}, {1, 1}, {2, 1},
  };

  for (const auto& [domain, range] : llvm::zip(expectedDomain, expectedRange)) {
    const auto& [row, col] = domain;
    const auto& [ct, slot] = range;
    auto maybeExists = actual.containsPointNoLocal({row, col, ct, slot});
    if (!maybeExists.has_value()) {
      FAIL() << "Failed to find point (" << row << ", " << col << ", " << ct
             << ", " << slot << ") in actual relation.";
    }
  }

  ASSERT_TRUE(actual.isEqual(expected));
}

TEST(HoistingTest, RowMajorLayoutRotateVecSlots) {
  MLIRContext context;
  // Same as DiagonalLayoutRotateVecSlots, but the matrix is laid out in
  // naive row-major order (one row per ciphertext)
  auto maybeFromVecLayout = getIntegerRelationFromIslStr(
      "{[d] -> [ct, slot] : (d - slot) mod 8 = 0 and d >= 0 and 7 >= d and "
      "slot >= 0 and 15 >= slot and ct = 0}");
  ASSERT_TRUE(succeeded(maybeFromVecLayout));
  auto fromVecLayout = maybeFromVecLayout.value();
  auto maybeToVecLayout = getIntegerRelationFromIslStr(
      "{[d] -> [ct, slot] : (d - (slot + 3)) mod 8 = 0 and d >= 0 and 7 >= d "
      "and slot >= 0 and 15 >= slot and ct = 0}");
  ASSERT_TRUE(succeeded(maybeToVecLayout));
  auto toVecLayout = maybeToVecLayout.value();
  auto maybeMatrixLayout = getIntegerRelationFromIslStr(
      "{[row, col] -> [ct, slot] : ct - row = 0 and (slot - col) mod 8 = 0 "
      "and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 15 >= slot and "
      "7 >= ct and 7 >= row and 7 >= col}");
  ASSERT_TRUE(succeeded(maybeMatrixLayout));
  auto matrixLayout = maybeMatrixLayout.value();
  const IntegerRelation expected =
      getIntegerRelationFromIslStr(
          "{[row, col] -> [ct, slot] : ct - row = 0 and "
          "((slot + 3) - col) mod 8 = 0 and row >= 0 and col >= 0 and ct >= "
          "0 and slot >= 0 and 15 >= slot and 7 >= ct and 7 >= row and 7 >= "
          "col}")
          .value();

  IntegerRelation actual =
      hoistConversionThroughMatvec(matrixLayout, fromVecLayout, toVecLayout);

  // Spot check some points
  std::vector<std::pair<int, int>> expectedDomain = {
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
  };
  std::vector<std::pair<int, int>> expectedRange = {
      {0, 5}, {1, 6}, {2, 7}, {3, 0}, {4, 0}, {4, 1}, {4, 2}, {4, 3},
  };

  for (const auto& [domain, range] : llvm::zip(expectedDomain, expectedRange)) {
    const auto& [row, col] = domain;
    const auto& [ct, slot] = range;
    auto maybeExists = actual.containsPointNoLocal({row, col, ct, slot});
    if (!maybeExists.has_value()) {
      FAIL() << "Failed to find point (" << row << ", " << col << ", " << ct
             << ", " << slot << ") in actual relation.";
    }
  }

  ASSERT_TRUE(actual.isEqual(expected));
}

TEST(HoistingTest, RowMajorLayoutExpandVecSlots) {
  MLIRContext context;
  // The vector packing changes from repeating the entire vec to repeating
  // individual entries. (e.g., a b c a b c -> a a b b c c)
  auto maybeFromVecLayout = getIntegerRelationFromIslStr(
      "{[d] -> [ct, slot] : (d - slot) mod 8 = 0 and d >= 0 and 7 >= d and "
      "slot >= 0 and 15 >= slot and ct = 0}");
  ASSERT_TRUE(succeeded(maybeFromVecLayout));
  auto fromVecLayout = maybeFromVecLayout.value();
  auto maybeToVecLayout = getIntegerRelationFromIslStr(
      "{[d] -> [ct, slot] : d - (slot // 2) = 0 and d >= 0 and 7 >= d "
      "and slot >= 0 and 15 >= slot and ct = 0}");
  ASSERT_TRUE(succeeded(maybeToVecLayout));
  auto toVecLayout = maybeToVecLayout.value();
  auto maybeMatrixLayout = getIntegerRelationFromIslStr(
      "{[row, col] -> [ct, slot] : ct - row = 0 and (slot - col) mod 8 = 0 "
      "and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 15 >= slot and "
      "7 >= ct and 7 >= row and 7 >= col}");
  ASSERT_TRUE(succeeded(maybeMatrixLayout));
  auto matrixLayout = maybeMatrixLayout.value();
  const IntegerRelation expected =
      getIntegerRelationFromIslStr(
          "{[row, col] -> [ct, slot] : ct - row = 0 and "
          "(slot // 2) - col = 0 and row >= 0 and col >= 0 and ct >= 0 "
          "and slot >= 0 and 15 >= slot and 7 >= ct and 7 >= row and 7 >= col}")
          .value();

  IntegerRelation actual =
      hoistConversionThroughMatvec(matrixLayout, fromVecLayout, toVecLayout);

  // Spot check some points
  std::vector<std::pair<int, int>> expectedDomain = {
      {0, 0}, {0, 0}, {1, 1}, {1, 1}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
  };
  std::vector<std::pair<int, int>> expectedRange = {
      {0, 0}, {0, 1}, {1, 2}, {1, 3}, {4, 6}, {4, 8}, {4, 10}, {4, 12},
  };

  for (const auto& [domain, range] : llvm::zip(expectedDomain, expectedRange)) {
    const auto& [row, col] = domain;
    const auto& [ct, slot] = range;
    auto maybeExists = actual.containsPointNoLocal({row, col, ct, slot});
    if (!maybeExists.has_value()) {
      FAIL() << "Failed to find point (" << row << ", " << col << ", " << ct
             << ", " << slot << ") in actual relation.";
    }
  }

  ASSERT_TRUE(actual.isEqual(expected));
}

TEST(HoistingTest, PushThroughInsertSlice) {
  // Insert a 4x4 slice (from a 32 slot ciphertext) into a 1x2x4x4 tensor. The
  // result tensor should have two ciphertexts.
  MLIRContext context;

  auto maybeSliceLayout = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 16 = 0 and "
      "0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }");
  ASSERT_TRUE(succeeded(maybeSliceLayout));

  auto actual = pushSliceLayoutThroughInsertSlice({1, 1, 4, 4}, {1, 2, 4, 4},
                                                  maybeSliceLayout.value());
  ASSERT_TRUE(succeeded(actual));

  // The result layout is a 4 dimension to ct, slot. Enumerate the points. Some
  // domain points can be mapped to many range points.
  PointPairCollector collector(4, 2);
  enumeratePoints(actual.value(), collector);
  std::map<std::vector<int64_t>, std::vector<std::vector<int64_t>>> pointMap;
  for (const auto& [domain, range] : collector.points) {
    if (!pointMap.contains(domain)) {
      pointMap[domain] = std::vector<std::vector<int64_t>>({range});
    }
    pointMap[domain].push_back(range);
  }

  std::vector<std::vector<int64_t>> expectedDomain = {
      {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 2, 1},
      {0, 1, 0, 0}, {0, 1, 1, 0}, {0, 1, 2, 1},
  };
  std::vector<std::vector<int64_t>> expectedRange = {
      {0, 0}, {0, 4}, {0, 9}, {1, 0}, {1, 4}, {1, 9},
  };

  for (const auto& [domain, range] : llvm::zip(expectedDomain, expectedRange)) {
    if (!pointMap.contains(domain)) {
      FAIL() << "Failed to find point (" << ::testing::PrintToString(domain)
             << ", " << ::testing::PrintToString(range)
             << ") in actual relation.";
    }
    if (!llvm::is_contained(pointMap[domain], range)) {
      FAIL() << "Expected point at " << ::testing::PrintToString(domain)
             << " to be " << ::testing::PrintToString(range) << ", got "
             << ::testing::PrintToString(pointMap[domain]);
    }
  }

  // There should be two ciphertexts in the result, each corresponding to two
  // slices in the second dimension (1x2x4x4).
  auto ctBound = actual.value().getConstantBound64(
      presburger::BoundType::UB,
      actual.value().getVarKindOffset(presburger::VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 1);  // inclusive
}

}  // namespace
}  // namespace heir
}  // namespace mlir
