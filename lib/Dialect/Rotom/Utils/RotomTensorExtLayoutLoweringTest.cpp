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
  // has one domain variable per tensor axis: i0 = row, i1 = col -- the pieces
  // of an axis share its variable rather than each binding their own. The
  // packing itself is plain 2x2-tiled row-major.
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

// The canonicalizing builder makes unused slot capacity explicit: a 4-vector
// at n = 8 gains a front gap piece, so the stored dims show every slot.
TEST(RotomTensorExtLayoutLoweringTest, BuilderInsertsExplicitGapFill) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d0});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/8);

  ASSERT_EQ(layout.getDims().size(), 2u);
  EXPECT_TRUE(cast<DimAttr>(layout.getDims()[0]).isGap());
  EXPECT_EQ(cast<DimAttr>(layout.getDims()[0]).getSize(), 2);

  FailureOr<LayoutData> data = preprocessLayoutAttr(layout);
  ASSERT_TRUE(succeeded(data));
  EXPECT_EQ(data->n, 8);
  EXPECT_EQ(data->ctPrefixLen, 0);
  ASSERT_EQ(data->pieces.size(), 2);
  EXPECT_EQ(data->pieces[0].kind, LayoutPieceKind::Gap);
  EXPECT_EQ(data->pieces[0].dim.getDim(), -2);
  EXPECT_EQ(data->pieces[0].dim.getSize(), 2);
  EXPECT_EQ(data->pieces[1].kind, LayoutPieceKind::Traversal);
}

TEST(RotomTensorExtLayoutLoweringTest, PreprocessSortsAxesByDimId) {
  // The deduped axes are canonicalized to ascending dim id regardless of
  // piece order, so the ISL lowering's domain variables always line up
  // positionally with tensor dims.
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  DimAttr d1 = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {d1, d0});
  LayoutAttr layout = LayoutAttr::get(&context, dims, /*n=*/16);

  FailureOr<LayoutData> data = preprocessLayoutAttr(layout);
  ASSERT_TRUE(succeeded(data));
  ASSERT_EQ(data->axes.size(), 2);
  EXPECT_EQ(data->axes[0].getDim(), 0);
  EXPECT_EQ(data->axes[1].getDim(), 1);
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

  // roll(1, 0) [R:2][0:8]: replica d of two holds the 8-vector rotated
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

  // roll(1, 0) [R:4][0:2]: shifts 2 and 3 wrap to 0 and 1.
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

// A plan-level pre-rotation shifts by multiplier * partner digit: replica d
// holds the vector rotated left by 4d -- the giant-step strides of a BSGS
// packing, composed into the relation without touching the layout's rolls.
TEST(RotomTensorExtLayoutLoweringTest,
     PreRotationMaterializesStridedRotations) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr repl = DimAttr::get(&context, /*dim=*/-1, /*size=*/4, /*stride=*/1);
  DimAttr d0 = DimAttr::get(&context, /*dim=*/0, /*size=*/16, /*stride=*/1);
  LayoutAttr layout =
      LayoutAttr::get(&context, ArrayAttr::get(&context, {repl, d0}), /*n=*/64);

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(
          layout, rotom::PreRotation{/*fromPiece=*/1, /*byPiece=*/0,
                                     /*multiplier=*/-4});
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<int> vec(16);
  for (int i = 0; i < 16; ++i) vec[i] = i + 1;
  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        return vec[domainPoint[0]];
      });
  ASSERT_EQ(packed.size(), 1u);
  ASSERT_EQ(packed[0].size(), 64u);
  for (int d = 0; d < 4; ++d) {
    for (int a = 0; a < 16; ++a) {
      EXPECT_EQ(packed[0][16 * d + a], vec[(a + 4 * d) % 16])
          << "replica " << d << " slot " << a;
    }
  }
}

// An `axis` FROM endpoint rewrites the whole dim's index, and each piece
// takes its digit of the rolled index: ciphertext (g, b) at slot a holds the
// iteration point k = (a + 4g + b) mod 16 -- the diagonal index split across
// two ciphertext digits (the borrow between digits is what a piece FROM
// cannot express).
TEST(RotomTensorExtLayoutLoweringTest, AxisRollOnSplitDim) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr kHi = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/4);
  DimAttr kLo = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  DimAttr i = DimAttr::get(&context, /*dim=*/0, /*size=*/16, /*stride=*/1);
  // rolls (axis 1, 2): the whole k index is rewritten to (k - i) mod 16;
  // k_hi emits its floor digit, k_lo its mod digit.
  LayoutAttr layout = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {kHi, kLo, i}), /*n=*/16,
      DenseI64ArrayAttr::get(
          &context, ArrayRef<int64_t>{
                        rotom::encodeRollEndpoint({/*isAxis=*/true, 1}), 2}));

  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        // f(i, k) = 16*i + k, distinct per iteration point.
        return 16 * static_cast<int>(domainPoint[0]) +
               static_cast<int>(domainPoint[1]);
      });
  ASSERT_EQ(packed.size(), 16u);
  for (int g = 0; g < 4; ++g) {
    for (int b = 0; b < 4; ++b) {
      for (int a = 0; a < 16; ++a) {
        const int k = (a + 4 * g + b) % 16;
        EXPECT_EQ(packed[4 * g + b][a], 16 * a + k)
            << "ct (" << g << ", " << b << ") slot " << a;
      }
    }
  }
}

// The full BSGS diagonal packing of a 16x16 matrix: the whole k axis rolled
// by i, then i rolled by the k_hi DIGIT of the rolled k with step -4.
// Ciphertext (g, b) at slot a holds the iteration point
// (i, k) = ((a - 4g) mod 16, (a + b) mod 16): the value the giant-step
// kernel multiplies against the baby-rotated vector block b, so that
// rotating partial sum g by 4g slots and adding yields the matvec.
TEST(RotomTensorExtLayoutLoweringTest, BsgsDiagonalPackingMaterializes) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr kHi = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/4);
  DimAttr kLo = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  DimAttr i = DimAttr::get(&context, /*dim=*/0, /*size=*/16, /*stride=*/1);
  LayoutAttr layout = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {kHi, kLo, i}), /*n=*/16,
      DenseI64ArrayAttr::get(
          &context, ArrayRef<int64_t>{
                        rotom::encodeRollEndpoint({/*isAxis=*/true, 1}), 2}));

  // The giant pre-rotation is the PLAN's encoding, not the layout's: piece 2
  // (i) shifts WITH the giant digit (piece 0) in strides of B = 4.
  FailureOr<std::string> isl =
      RotomTensorExtLayoutLowering::lowerToTensorExtIsl(
          layout, rotom::PreRotation{/*fromPiece=*/2, /*byPiece=*/0,
                                     /*multiplier=*/4});
  ASSERT_TRUE(succeeded(isl));
  auto relation = getIntegerRelationFromIslStr(*isl);
  ASSERT_TRUE(succeeded(relation));

  std::vector<std::vector<int>> packed = evaluateLayout<int>(
      relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
        return 16 * static_cast<int>(domainPoint[0]) +
               static_cast<int>(domainPoint[1]);
      });
  ASSERT_EQ(packed.size(), 16u);
  for (int g = 0; g < 4; ++g) {
    for (int b = 0; b < 4; ++b) {
      for (int a = 0; a < 16; ++a) {
        const int iVal = ((a - 4 * g) % 16 + 16) % 16;
        const int kVal = (a + b) % 16;
        EXPECT_EQ(packed[4 * g + b][a], 16 * iVal + kVal)
            << "ct (" << g << ", " << b << ") slot " << a;
      }
    }
  }
}

// A piece FROM on a split dim rewrites only that piece's digit -- no borrow
// crosses into the other digits. Rolling the high digit shifts the axis in
// strides of 4; rolling the low digit rotates within each block of 4.
TEST(RotomTensorExtLayoutLoweringTest, PieceRollOnSplitDimRotatesOneDigit) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr repl = DimAttr::get(&context, /*dim=*/-1, /*size=*/4, /*stride=*/1);
  DimAttr d0Hi = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/4);
  DimAttr d0Lo = DimAttr::get(&context, /*dim=*/0, /*size=*/4, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {repl, d0Hi, d0Lo});
  std::vector<int> vec(16);
  for (int i = 0; i < 16; ++i) vec[i] = i + 1;
  auto evaluate = [&](LayoutAttr layout) {
    FailureOr<std::string> isl =
        RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
    EXPECT_TRUE(succeeded(isl));
    auto relation = getIntegerRelationFromIslStr(*isl);
    EXPECT_TRUE(succeeded(relation));
    return evaluateLayout<int>(
        relation.value(), [&](const std::vector<int64_t>& domainPoint) -> int {
          return vec[domainPoint[0]];
        });
  };

  // roll(1, 0): the high digit shifts by the replica index, so replica d
  // holds the vector rotated by 4d whole (the low bits ride along
  // untouched, so no borrow is observable here).
  LayoutAttr hiRolled = LayoutAttr::get(
      &context, dims, /*n=*/64,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{1, 0}));
  std::vector<std::vector<int>> packed = evaluate(hiRolled);
  ASSERT_EQ(packed.size(), 1u);
  ASSERT_EQ(packed[0].size(), 64u);
  for (int d = 0; d < 4; ++d) {
    for (int a = 0; a < 16; ++a) {
      EXPECT_EQ(packed[0][16 * d + a], vec[(a + 4 * d) % 16])
          << "hi-rolled replica " << d << " slot " << a;
    }
  }

  // roll(2, 0): the LOW digit shifts by the replica index -- each block of
  // 4 rotates independently, with no borrow into the high digit (the
  // whole-axis roll, spelled `axis 0`, would borrow).
  LayoutAttr loRolled = LayoutAttr::get(
      &context, dims, /*n=*/64,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{2, 0}));
  packed = evaluate(loRolled);
  ASSERT_EQ(packed.size(), 1u);
  for (int d = 0; d < 4; ++d) {
    for (int a = 0; a < 16; ++a) {
      const int block = a / 4;
      const int within = (a % 4 + d) % 4;
      EXPECT_EQ(packed[0][16 * d + a], vec[4 * block + within])
          << "lo-rolled replica " << d << " slot " << a;
    }
  }
}

// Axis endpoints: legal only on a split axis (the piece spelling is
// canonical when the axis is one piece), and a roll may not shift an axis by
// one of its own pieces.
TEST(RotomTensorExtLayoutLoweringTest, AxisRollEndpointVerifierRules) {
  MLIRContext context;
  context.loadDialect<RotomDialect>();
  DimAttr kHi = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/4);
  DimAttr kLo = DimAttr::get(&context, /*dim=*/1, /*size=*/4, /*stride=*/1);
  DimAttr i = DimAttr::get(&context, /*dim=*/0, /*size=*/16, /*stride=*/1);
  ArrayAttr dims = ArrayAttr::get(&context, {kHi, kLo, i});
  ScopedDiagnosticHandler silence(&context,
                                  [](Diagnostic&) { return success(); });
  auto swallow =
      mlir::detail::getDefaultDiagnosticEmitFn(UnknownLoc::get(&context));
  const int64_t axisK = rotom::encodeRollEndpoint({/*isAxis=*/true, 1});
  const int64_t axisI = rotom::encodeRollEndpoint({/*isAxis=*/true, 0});
  const int64_t axisMissing = rotom::encodeRollEndpoint({/*isAxis=*/true, 5});

  // FROM the split axis, BY the unsplit i piece: legal.
  EXPECT_TRUE(succeeded(LayoutAttr::verify(
      swallow, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{axisK, 2}))));
  // An axis endpoint on an unsplit axis must use the piece spelling.
  EXPECT_TRUE(failed(LayoutAttr::verify(
      swallow, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{axisI, 0}))));
  // An axis endpoint must name an axis present in dims.
  EXPECT_TRUE(failed(LayoutAttr::verify(
      swallow, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{axisMissing, 2}))));
  // An axis may not be shifted by one of its own pieces.
  EXPECT_TRUE(failed(LayoutAttr::verify(
      swallow, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{axisK, 0}))));
  // ... nor may a piece be shifted by its own whole axis.
  EXPECT_TRUE(failed(LayoutAttr::verify(
      swallow, dims, /*n=*/16,
      DenseI64ArrayAttr::get(&context, ArrayRef<int64_t>{0, axisK}))));
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

// The rollSteps storage must be the canonical non-trivial form so identical
// layouts unique to identical attributes: an all-unit or (non-null) empty
// steps array must be omitted, not stored.
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
