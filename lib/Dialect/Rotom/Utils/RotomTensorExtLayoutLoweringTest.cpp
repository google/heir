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
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
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
        // Domain vars are tensor dims in order: [row, col].
        return matrix[domainPoint[0]][domainPoint[1]];
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
  // dim ids 0 and 1 are each split mixed-radix into two pieces, so the relation
  // has one domain variable per tensor axis: i0 = row, i1 = col. The packing is
  // unchanged (2x2-tiled row-major) -- only the domain representation differs
  // from the old per-piece variables.
  ASSERT_EQ(relation->getNumDomainVars(), 2);
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        return matrix[domainPoint[0]][domainPoint[1]];
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
        // Domain vars are tensor dims in order: [row, col].
        return matrix[domainPoint[0]][domainPoint[1]];
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

TEST(RotomTensorExtLayoutLoweringTest, PreprocessSortsTraversalDimsByDimId) {
  // The deduped traversal dims are canonicalized to ascending dim id
  // regardless of piece order, so the ISL lowering's domain variables always
  // line up positionally with tensor dims.
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d1, d0});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/16);

  FailureOr<LayoutData> data = preprocessLayoutAttr(layout);
  ASSERT_TRUE(succeeded(data));
  ASSERT_EQ(data->traversalDims.size(), 2);
  EXPECT_EQ(data->traversalDims[0].getDim(), 0);
  EXPECT_EQ(data->traversalDims[1].getDim(), 1);
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

// roll(1, 0) with piece 0 a replication dim: dim 0's index is shifted by the
// replica index, so replica d holds the vector rotated left by d -- the
// layout materializes every cyclic rotation, making downstream alignment a
// replica selection.
TEST(RotomTensorExtLayoutLoweringTest, RollByReplicationMaterializesRotations) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr repl = DimAttr::get(&context, /*dim=*/-1, /*size=*/4, /*stride=*/1);
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {repl, d0});
  LayoutAttr layout = LayoutAttr::get(
      &context, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}));

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<int> vec = {1, 2, 3, 4};
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        return vec[domainPoint[0]];
      });
  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 2, 4, 1, 2, 3}};
  EXPECT_EQ(packed, expected);
}

// Unequal roll extents: the shift reduces modulo the rolled dim's extent.
// A smaller partner materializes a prefix of the rotations; a larger one
// wraps around.
TEST(RotomTensorExtLayoutLoweringTest, UnequalExtentRollReducesModFromSize) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();

  // [R:2][0:8], roll(1, 0): replica d of two holds the 8-vector rotated
  // left by d.
  DimAttr repl2 = DimAttr::get(&context, /*dim=*/-1, /*size=*/2, /*stride=*/1);
  DimAttr d0Of8 = DimAttr::get(&context, /*dim=*/0, /*size=*/8, /*stride=*/1);
  LayoutAttr prefix = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {repl2, d0Of8}), /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}));
  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(prefix);
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        return vec[domainPoint[0]];
      });
  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 1}};
  EXPECT_EQ(packed, expected);

  // [R:4][0:2], roll(1, 0): shifts 2 and 3 wrap to 0 and 1.
  DimAttr repl4 = DimAttr::get(&context, /*dim=*/-1, /*size=*/4, /*stride=*/1);
  DimAttr d0Of2 = DimAttr::get(&context, /*dim=*/0, /*size=*/2, /*stride=*/1);
  LayoutAttr wrap = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {repl4, d0Of2}), /*n=*/8,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}));
  isl = RotomTensorExtLayoutLowering::lowerToTensorExtIsl(wrap);
  ASSERT_TRUE(succeeded(isl));
  relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));
  std::vector<int> pair = {1, 2};
  packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        return pair[domainPoint[0]];
      });
  expected = {{1, 2, 2, 1, 1, 2, 2, 1}};
  EXPECT_EQ(packed, expected);
}

TEST(RotomTensorExtLayoutLoweringTest, RollVerifierDimKindRules) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr repl = DimAttr::get(&context, /*dim=*/-1, /*size=*/4, /*stride=*/1);
  DimAttr gap = DimAttr::get(&context, /*dim=*/-2, /*size=*/4, /*stride=*/1);
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  ScopedDiagnosticHandler silence(&context,
                                  [](Diagnostic&) { return success(); });
  auto swallow =
      mlir::detail::getDefaultDiagnosticEmitFn(UnknownLoc::get(&context));

  // Rolling a traversal dim BY a replication dim of equal extent is allowed.
  ArrayAttr dims = ArrayAttr::get(&context, {repl, d0});
  EXPECT_TRUE(succeeded(LayoutAttr::verify(
      swallow, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}))));
  // Rolling FROM a replication dim stays rejected (no index to rewrite).
  EXPECT_TRUE(failed(LayoutAttr::verify(
      swallow, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{0, 1}))));
  // Rolling BY a gap dim of equal extent is allowed; rolling FROM a gap dim
  // stays rejected.
  ArrayAttr gapDims = ArrayAttr::get(&context, {gap, d0});
  EXPECT_TRUE(succeeded(LayoutAttr::verify(
      swallow, gapDims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}))));
  EXPECT_TRUE(failed(LayoutAttr::verify(
      swallow, gapDims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{0, 1}))));
}

// roll(1, 0) with piece 0 a gap dim: the gap's block index is the shift, so
// block g holds the vector rotated left by g -- a rolled-by gap claims its
// blocks with the rotations (a plain gap would leave them unclaimed).
TEST(RotomTensorExtLayoutLoweringTest, RollByGapMaterializesRotations) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr gap = DimAttr::get(&context, /*dim=*/-2, /*size=*/4, /*stride=*/1);
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {gap, d0});
  LayoutAttr layout = LayoutAttr::get(
      &context, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}));

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<int> vec = {1, 2, 3, 4};
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        return vec[domainPoint[0]];
      });
  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 2, 4, 1, 2, 3}};
  EXPECT_EQ(packed, expected);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
