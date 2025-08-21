#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Parser.h"
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
  IntegerRelation fromVecLayout = relationFromString(
      "(d, ct, slot) : "
      "((d - slot) mod 8 == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation toVecLayout = relationFromString(
      "(d, ct, slot) : "
      "((d - (slot + 3)) mod 8 == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation matrixLayout = relationFromString(
      "(row, col, ct, slot) : "
      "((slot - row) mod 8 == 0, "
      "(ct + slot - col) mod 8 == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);
  const IntegerRelation expected = relationFromString(
      "(row, col, ct, slot) : "
      "(((slot + 3) - row) mod 8 == 0, "
      "(ct + (slot + 3) - col) mod 8 == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);

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
  IntegerRelation fromVecLayout = relationFromString(
      "(d, ct, slot) : "
      "((d - slot) mod 8 == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation toVecLayout = relationFromString(
      "(d, ct, slot) : "
      "((d - (slot + 3)) mod 8 == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation matrixLayout = relationFromString(
      "(row, col, ct, slot) : "
      "(ct - row == 0, "
      "(slot - col) mod 8 == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);
  const IntegerRelation expected = relationFromString(
      "(row, col, ct, slot) : "
      "(ct - row == 0, "
      "((slot + 3) - col) mod 8 == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);

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
  IntegerRelation fromVecLayout = relationFromString(
      "(d, ct, slot) : "
      "((d - slot) mod 8 == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation toVecLayout = relationFromString(
      "(d, ct, slot) : "
      "(d - (slot floordiv 2) == 0, "
      "d >= 0, 7 >= d, slot >= 0, 15 >= slot, ct == 0)",
      1, &context);
  IntegerRelation matrixLayout = relationFromString(
      "(row, col, ct, slot) : "
      "(ct - row == 0, "
      "(slot - col) mod 8 == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);
  const IntegerRelation expected = relationFromString(
      "(row, col, ct, slot) : "
      "(ct - row == 0, "
      "(slot floordiv 2) - col == 0, "
      "row >= 0, col >= 0, ct >= 0, slot >= 0, "
      "15 >= slot, 7 >= ct, 7 >= row, 7 >= col)",
      2, &context);

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

}  // namespace
}  // namespace heir
}  // namespace mlir
