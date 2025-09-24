#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/IslConversion.h"
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
using presburger::VarKind;

void runRowMajorTest(RankedTensorType tensorType, int64_t numSlots) {
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

  runRowMajorTest(tensorType, numSlots);
}

TEST(UtilsTest, TwoCiphertexts) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({4}, IndexType::get(&context));
  int64_t numSlots = 2;
  runRowMajorTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDim) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 8;
  runRowMajorTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDimSingleCiphertext) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 24;
  runRowMajorTest(tensorType, numSlots);
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

TEST(UtilsTest, TestGetRangePoints) {
  MLIRContext context;
  IntegerRelation rel =
      relationFromString("(x) : (x >= 0, 7 >= x, x mod 3 == 0)", 0, &context);
  std::vector<std::vector<int64_t>> expected = {{0}, {3}, {6}};
  PointCollector collector;
  getRangePoints(rel, collector);
  EXPECT_EQ(collector.points, expected);
}

TEST(UtilsTest, TestEnumeratePoints) {
  MLIRContext context;
  // Create a relation with 1 domain variable (x) and 1 range variable (y)
  IntegerRelation rel = relationFromString(
      "(x, y) : (x >= 0, 2 >= x, y >= 0, 1 >= y)", 1, &context);
  PointPairCollector collector(1, 1);  // 1 domain dim, 1 range dim
  enumeratePoints(rel, collector);

  // Expected points: domain x range pairs for x in [0,2] and y in [0,1]
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> expected =
      {{{0}, {0}}, {{0}, {1}}, {{1}, {0}}, {{1}, {1}}, {{2}, {0}}, {{2}, {1}}};

  EXPECT_EQ(collector.points.size(), expected.size());
  for (const auto& expectedPoint : expected) {
    bool found = false;
    for (const auto& actualPoint : collector.points) {
      if (actualPoint.first == expectedPoint.first &&
          actualPoint.second == expectedPoint.second) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Expected point not found: domain="
                       << expectedPoint.first[0]
                       << ", range=" << expectedPoint.second[0];
  }
}

TEST(UtilsTest, PerRowLayout) {
  MLIRContext context;

  // Per row layout 3x5 matrix
  //  1  2  3  4  5
  //  6  7  8  9 10
  // 11 12 13 14 15
  // to
  //  1  2  3  4  5 * * *  1  2  3  4  5 * * *
  //  6  7  8  9 10 * * *  6  7  8  9 10 * * *
  // 11 12 13 14 15 * * * 11 12 13 14 15 * * *
  int64_t ciphertextSize = 16;
  RankedTensorType matrixType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation perRowRelation =
      getPerRowLayoutRelation(matrixType, ciphertextSize);
  int64_t paddedCols = 8;

  for (unsigned int i = 0; i < 3; ++i) {
    for (unsigned int j = 0; j < 16; ++j) {
      auto row = i;
      auto col = j % paddedCols;
      if (col >= matrixType.getDimSize(1)) {
        EXPECT_FALSE(
            perRowRelation.containsPointNoLocal({row, col, i, j}).has_value());
      } else {
        auto maybeExists =
            perRowRelation.containsPointNoLocal({row, col, i, j});
        EXPECT_TRUE(maybeExists.has_value());
      }
    }
  }
}

TEST(UtilsTest, TestAnyRangePoint) {
  MLIRContext context;
  IntegerRelation rel =
      relationFromString("(x) : (x >= 0, 7 >= x, x mod 3 == 0)", 0, &context);
  std::vector<int64_t> actual = anyRangePoint(rel);
  EXPECT_TRUE(rel.containsPointNoLocal(actual).has_value());
}

TEST(UtilsTest, ConvFilterRelation) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  int64_t padding = 1;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 8);

  // Handwritten expected relation
  auto relation = getIntegerRelationFromIslStr(
      "{ [ifr, ifc] -> [mr, mc] : exists idr, idc : -1 <= idr and idr <= 1 and "
      "-1 <= idc and idc <= 1 and 0 <= ifr and ifr <= 2 and 0 <= ifc and ifc "
      "<= 2 and mr = idc + 1 + 3 * (idr + 1) and mc = -4 + mr + ifc + "
      "ifr * 3 and 0 <= idr + ifr and idr + ifr <= 2 and 0 <= idc + ifc and "
      "idc + ifc <= 2 }");
  relation.value().simplify();
  ASSERT_TRUE(succeeded(relation));
  EXPECT_TRUE(convFilterRelation.isEqual(relation.value()));
}

TEST(UtilsTest, ConvFilterRelationNoPadding) {
  // No padding and same size should result in a single multiplication of the
  // two flattened inputs.
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  int64_t padding = 0;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 0);
}

TEST(UtilsTest, ConvFilterRelation4x4Data) {
  // No padding on a larger data matrix should result in 4 ciphertexts.
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({4, 4}, IndexType::get(&context));
  int64_t padding = 0;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 3);
}

TEST(UtilsTest, ConvFilterRelationPadding2) {
  // Two padding on a larger data matrix should result in 36 rows.
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({4, 4}, IndexType::get(&context));
  int64_t padding = 2;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 35);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
