#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::LayoutData;
using rotom::LayoutPieceKind;
using rotom::preprocessLayoutAttr;
using rotom::RotomDialect;
using rotom::RotomTensorExtLayoutLowering;

FailureOr<presburger::IntegerRelation> lowerToRelation(LayoutAttr layout) {
  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  if (failed(isl)) return failure();
  return getIntegerRelationFromIslStr(*isl);
}

std::vector<std::vector<int64_t>> rangePointsForDomain(
    const presburger::IntegerRelation& relation, ArrayRef<int64_t> domain) {
  PointCollector collector;
  getRangePoints(fixDomainVars(relation, domain), collector);
  return collector.points;
}

TEST(RotomTensorExtLayoutLoweringTest, RowMajor4x4Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0, d1});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/16);

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));

  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation.value(), matrix);

  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
  };
  EXPECT_EQ(packed, expected);
}

TEST(RotomTensorExtLayoutLoweringTest, ColumnMajor4x4Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d1, d0});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/16);

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));

  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        // Traversal dims are {dim1, dim0}, so relation vars are [col, row].
        return matrix[domainPoint[1]][domainPoint[0]];
      });

  std::vector<std::vector<int>> expected = {
      {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16},
  };
  EXPECT_EQ(packed, expected);
}

TEST(RotomTensorExtLayoutLoweringTest, TiledRowMajor4x4Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/2, /*stride=*/2);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/2, /*stride=*/2);
  DimAttr d2 = DimAttr::get(&context, /*dim=*/0, /*size=*/2, /*stride=*/1);
  DimAttr d3 = DimAttr::get(&context, /*dim=*/1, /*size=*/2, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0, d1, d2, d3});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/16);

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));

  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  ASSERT_EQ(relation->getNumDomainVars(), 4);
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        const int64_t row = domainPoint[0] * 2 + domainPoint[2];
        const int64_t col = domainPoint[1] * 2 + domainPoint[3];
        return matrix[row][col];
      });

  std::vector<std::vector<int>> expected = {
      {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16},
  };
  EXPECT_EQ(packed, expected);
}

TEST(RotomTensorExtLayoutLoweringTest, SplitRowMajor4x4Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0, d1});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/4);

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));

  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation.value(), matrix);

  // Row-major packing, split into 4 ciphertexts of 4 slots: one row per
  // ciphertext.
  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  EXPECT_EQ(packed, expected);
}

TEST(RotomTensorExtLayoutLoweringTest, SplitColumnMajor4x4Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0, d1});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/4);

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));

  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        // Traversal dims are {dim1, dim0}, so relation vars are [col, row].
        return matrix[domainPoint[1]][domainPoint[0]];
      });

  // Column-major packing, split into 4 ciphertexts of 4 slots: one column per
  // ciphertext.
  std::vector<std::vector<int>> expected = {
      {1, 5, 9, 13},
      {2, 6, 10, 14},
      {3, 7, 11, 15},
      {4, 8, 12, 16},
  };
  EXPECT_EQ(packed, expected);
}

TEST(RotomTensorExtLayoutLoweringTest,
     RotomMatmulLoweringReferenceMatchesRectangularMatmul) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr lhsM = DimAttr::get(&context, /*dim=*/0, /*size=*/2, /*stride=*/1);
  DimAttr lhsK = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  DimAttr rhsK = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr rhsN = DimAttr::get(&context, /*dim=*/1, /*size=*/2, /*stride=*/1);
  DimAttr outM = DimAttr::get(&context, /*dim=*/0, /*size=*/2, /*stride=*/1);
  DimAttr outN = DimAttr::get(&context, /*dim=*/1, /*size=*/2, /*stride=*/1);

  LayoutAttr lhsLayout =
      LayoutAttr::get(&context, ArrayAttr::get(&context, {lhsM, lhsK}),
                      /*n=*/8);
  LayoutAttr rhsLayout =
      LayoutAttr::get(&context, ArrayAttr::get(&context, {rhsK, rhsN}),
                      /*n=*/8);
  LayoutAttr resultLayout =
      LayoutAttr::get(&context, ArrayAttr::get(&context, {outM, outN}),
                      /*n=*/8);

  FailureOr<presburger::IntegerRelation> lhsRelation =
      lowerToRelation(lhsLayout);
  ASSERT_TRUE(succeeded(lhsRelation));
  FailureOr<presburger::IntegerRelation> rhsRelation =
      lowerToRelation(rhsLayout);
  ASSERT_TRUE(succeeded(rhsRelation));
  FailureOr<presburger::IntegerRelation> resultRelation =
      lowerToRelation(resultLayout);
  ASSERT_TRUE(succeeded(resultRelation));

  std::vector<std::vector<int>> lhs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
  };
  std::vector<std::vector<int>> rhs = {
      {2, 3},
      {5, 7},
      {11, 13},
      {17, 19},
  };
  std::vector<std::vector<int>> init = {
      {100, 200},
      {300, 400},
  };

  std::vector<std::vector<int>> lhsPacked =
      evaluateLayoutOnMatrix(*lhsRelation, lhs);
  std::vector<std::vector<int>> rhsPacked =
      evaluateLayoutOnMatrix(*rhsRelation, rhs);
  std::vector<std::vector<int>> acc =
      evaluateLayoutOnMatrix(*resultRelation, init);

  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        std::vector<std::vector<int64_t>> lhsPoints =
            rangePointsForDomain(*lhsRelation, {i, k});
        std::vector<std::vector<int64_t>> rhsPoints =
            rangePointsForDomain(*rhsRelation, {k, j});
        std::vector<std::vector<int64_t>> resultPoints =
            rangePointsForDomain(*resultRelation, {i, j});
        ASSERT_EQ(lhsPoints.size(), 1);
        ASSERT_EQ(rhsPoints.size(), 1);
        ASSERT_EQ(resultPoints.size(), 1);

        const std::vector<int64_t>& lhsPoint = lhsPoints[0];
        const std::vector<int64_t>& rhsPoint = rhsPoints[0];
        const std::vector<int64_t>& resultPoint = resultPoints[0];
        acc[resultPoint[0]][resultPoint[1]] +=
            lhsPacked[lhsPoint[0]][lhsPoint[1]] *
            rhsPacked[rhsPoint[0]][rhsPoint[1]];
      }
    }
  }

  std::vector<std::vector<int>> actual =
      unpackLayoutToMatrix(*resultRelation, acc, {2, 2});
  std::vector<std::vector<int>> expected = {
      {100 + 1 * 2 + 2 * 5 + 3 * 11 + 4 * 17,
       200 + 1 * 3 + 2 * 7 + 3 * 13 + 4 * 19},
      {300 + 5 * 2 + 6 * 5 + 7 * 11 + 8 * 17,
       400 + 5 * 3 + 6 * 7 + 7 * 13 + 8 * 19},
  };
  EXPECT_EQ(actual, expected);
}

TEST(RotomTensorExtLayoutLoweringTest, PreprocessAddsImplicitGap) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/8);

  FailureOr<LayoutData> data = preprocessLayoutAttr(layout);
  ASSERT_TRUE(succeeded(data));
  EXPECT_EQ(data->n, 8);
  EXPECT_EQ(data->ctPrefixLen, 0);
  ASSERT_EQ(data->gapDims.size(), 1);
  EXPECT_EQ(data->gapDims[0].getDim(), -2);
  EXPECT_EQ(data->gapDims[0].getSize(), 2);
  ASSERT_EQ(data->pieces.size(), 2);
  EXPECT_EQ(data->pieces[0], LayoutPieceKind::Gap);
  EXPECT_EQ(data->pieces[1], LayoutPieceKind::Traversal);
}

TEST(RotomTensorExtLayoutLoweringTest, PreprocessPreservesTraversalDimsOrder) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d1, d0});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/16);

  FailureOr<LayoutData> data = preprocessLayoutAttr(layout);
  ASSERT_TRUE(succeeded(data));
  ASSERT_EQ(data->traversalDims.size(), 2);
  EXPECT_EQ(data->traversalDims[0].getDim(), 1);
  EXPECT_EQ(data->traversalDims[1].getDim(), 0);
}

TEST(RotomTensorExtLayoutLoweringTest, RolledRowMajor2x2Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/2, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/2, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0, d1});
  LayoutAttr layout = LayoutAttr::get(
      &context, dims, /*n=*/4,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{0, 1}));

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2},
      {3, 4},
  };
  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation.value(), matrix);
  std::vector<std::vector<int>> expected = {{1, 4, 3, 2}};
  EXPECT_EQ(packed, expected);
  EXPECT_EQ(unpackLayoutToMatrix(relation.value(), packed, {2, 2}), matrix);
}

TEST(RotomTensorExtLayoutLoweringTest, RolledRowMajor4x4Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0, d1});
  LayoutAttr layout = LayoutAttr::get(
      &context, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{0, 1}));

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation.value(), matrix);

  // ``roll(0,1)``: diagonal ``(i0 - i1) mod 4`` classes, listed in Rotom order.
  std::vector<std::vector<int>> expected = {
      {1, 6, 11, 16, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12},
  };
  EXPECT_EQ(packed, expected);
  EXPECT_EQ(unpackLayoutToMatrix(relation.value(), packed, {4, 4}), matrix);
}

TEST(RotomTensorExtLayoutLoweringTest, RolledInternalRowMajor4x4Evaluate) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0, d1});
  LayoutAttr layout = LayoutAttr::get(
      &context, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}));

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation.value(), matrix);

  // ``roll(1,0)``: cyclic column order within each row (dims index high first).
  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4, 6, 7, 8, 5, 11, 12, 9, 10, 16, 13, 14, 15},
  };
  EXPECT_EQ(packed, expected);
  EXPECT_EQ(unpackLayoutToMatrix(relation.value(), packed, {4, 4}), matrix);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
