#include <cmath>
#include <cstdint>
#include <optional>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Parser.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

void runTest(RankedTensorType tensorType, int64_t numSlots) {
  IntegerRelation result = getRowMajorLayoutRelation(tensorType, numSlots);

  // Check that the result relation requires size(tensor) / slots ciphertexts.
  auto ctIndex = result.getVarKindOffset(VarKind::Range);
  std::optional<int64_t> numCiphertexts =
      result.getConstantBound64(BoundType::UB, ctIndex);
  ASSERT_TRUE(numCiphertexts.has_value());
  EXPECT_EQ(numCiphertexts.value(),
            std::ceil(tensorType.getNumElements() / (double)numSlots) - 1);

  // Ensure that the layout is row-major.
  SmallVector<int64_t> shape = llvm::to_vector(tensorType.getShape());
  for (int64_t i = 0; i < tensorType.getNumElements(); ++i) {
    SmallVector<int64_t> indices = getIndicesFromRowMajorShape(i, shape);
    indices.push_back(static_cast<int64_t>(std::floor(i / (double)numSlots)));
    indices.push_back(i % numSlots);
    auto maybeExists = result.containsPointNoLocal(indices);
    EXPECT_TRUE(maybeExists.has_value());
  }
}

TEST(UtilsTest, TestAddModConstraint) {
  MLIRContext context;

  IntegerRelation rel =
      relationFromString("(x) : (x >= 0, 100 - x >= 0)", 1, &context);
  unsigned result = addModConstraint(rel, {1, 0}, 32);  // x % 32
  rel.convertVarKind(VarKind::Local,
                     result - rel.getVarKindOffset(VarKind::Local),
                     rel.getNumVarKind(VarKind::Local), VarKind::Range);
  for (unsigned x = 0; x <= 100; ++x) {
    EXPECT_TRUE(rel.containsPointNoLocal({x, x % 32}));
  }
}

TEST(UtilsTest, SingleCiphertext) {
  // Add row major layout relation when number of slots is exactly the number of
  // elements.
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2}, IndexType::get(&context));
  int64_t numSlots = tensorType.getNumElements();

  runTest(tensorType, numSlots);
}

TEST(UtilsTest, TwoCiphertexts) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({4}, IndexType::get(&context));
  int64_t numSlots = 2;
  runTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDim) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 8;
  runTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDimSingleCiphertext) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 24;
  runTest(tensorType, numSlots);
}

TEST(UtilsTest, DiagonalLayout) {
  MLIRContext context;

  // Diagonalize a 4x8 matrix into a 4x64 matrix.
  int64_t ciphertextSize = 64;
  RankedTensorType matrixType =
      RankedTensorType::get({4, 8}, IndexType::get(&context));
  IntegerRelation diagonalRelation =
      getDiagonalLayoutRelation(matrixType, ciphertextSize);

  diagonalRelation.simplify();
  for (unsigned int i = 0; i < 4; ++i) {
    for (unsigned int j = 0; j < 64; ++j) {
      auto maybeExists =
          diagonalRelation.containsPointNoLocal({j % 4, (i + j) % 8, i, j});
      EXPECT_TRUE(maybeExists.has_value());
    }
  }
}

TEST(UtilsTest, SquatDiagonalLayout) {
  MLIRContext context;

  // Diagonalize a 3x5 matrix - this will require padding the row to 4 and the
  // cols to 8
  //
  //  1  2  3  4  5  *  *  *
  //  6  7  8  9 10  *  *  *
  // 11 12 13 14 15  *  *  *
  //  *  *  *  *  *  *  *  *

  // 1  7 13  * 5 *  * *
  // 2  8 14  * * *  * *
  // 3  9 15  * * * 11 *
  // 4 10  *  * * 6 12 *
  int64_t ciphertextSize = 8;
  RankedTensorType matrixType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation diagonalRelation =
      getDiagonalLayoutRelation(matrixType, ciphertextSize);
  int64_t paddedRows = 4;
  int64_t paddedCols = 8;

  for (unsigned int i = 0; i < 4; ++i) {
    for (unsigned int j = 0; j < 8; ++j) {
      auto row = j % paddedRows;
      auto col = (i + j) % paddedCols;
      if (row >= matrixType.getDimSize(0) || col >= matrixType.getDimSize(1)) {
        EXPECT_FALSE(diagonalRelation.containsPointNoLocal({row, col, i, j})
                         .has_value());
      } else {
        auto maybeExists =
            diagonalRelation.containsPointNoLocal({row, col, i, j});
        EXPECT_TRUE(maybeExists.has_value());
      }
    }
  }
}

TEST(UtilsTest, TestGetPointsInRelation) {
  MLIRContext context;
  IntegerRelation rel =
      relationFromString("(x) : (x >= 0, 7 >= x, x mod 3 == 0)", 0, &context);
  std::vector<std::vector<int>> expected = {{0}, {3}, {6}};
  PointCollector collector;
  getRangePoints(rel, collector);
  EXPECT_EQ(collector.points, expected);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
