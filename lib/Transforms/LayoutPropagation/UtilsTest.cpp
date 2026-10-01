#include <cstdint>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Transforms/LayoutPropagation/Utils.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using llvm::SmallVector;

TEST(UtilsTest, TestShiftByInserted1) {
  SmallVector<int64_t> dims = {0, 1, 2, 3};
  SmallVector<int64_t> inserts = {1, 2};
  SmallVector<int64_t> expected = {0, 3, 4, 5};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByInserted2) {
  SmallVector<int64_t> dims = {2, 6, 7, 8};
  SmallVector<int64_t> inserts = {0, 4};
  SmallVector<int64_t> expected = {3, 8, 9, 10};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByInsertedCollision) {
  SmallVector<int64_t> dims = {3, 6, 7, 8};
  SmallVector<int64_t> inserts = {0, 4};
  SmallVector<int64_t> expected = {5, 8, 9, 10};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemoved1) {
  SmallVector<int64_t> dims = {0, 3, 4, 5};
  SmallVector<int64_t> removals = {1, 2};
  SmallVector<int64_t> expected = {0, 1, 2, 3};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemoved2) {
  SmallVector<int64_t> dims = {3, 8, 9, 10};
  SmallVector<int64_t> removals = {0, 4};
  SmallVector<int64_t> expected = {2, 6, 7, 8};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemovedCollision) {
  SmallVector<int64_t> dims = {5, 8, 9, 10};
  SmallVector<int64_t> removals = {0, 4};
  SmallVector<int64_t> expected = {3, 6, 7, 8};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestReduceLayout) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 4x6 tensor packed into a 3x8 tensor along dimension 0.
  RankedTensorType tensorType =
      RankedTensorType::get({4, 6}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  LayoutAttr layout = LayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {0};
  LayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 1);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  presburger::IntegerRelation expectedRelation = layout.getIntegerRelation();
  expectedRelation.projectOut(0, 1);
  expectedRelation =
      LayoutAttr::getFromIntegerRelation(&context, expectedRelation)
          .getIntegerRelation();

  EXPECT_TRUE(isRelationEqual(reducedRelation, expectedRelation));
}

TEST(UtilsTest, TestReduceLayoutMultiDim) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 3x2x4 tensor packed into a 3x8 tensor along dimension 2.
  RankedTensorType tensorType =
      RankedTensorType::get({3, 2, 4}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  LayoutAttr layout = LayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {2};
  LayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 2);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  presburger::IntegerRelation expectedRelation = layout.getIntegerRelation();
  expectedRelation.projectOut(2, 1);
  expectedRelation =
      LayoutAttr::getFromIntegerRelation(&context, expectedRelation)
          .getIntegerRelation();

  EXPECT_TRUE(isRelationEqual(reducedRelation, expectedRelation));
}

TEST(UtilsTest, TestReduceLayoutManyReductions) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 3x2x4 tensor packed into a 3x8 tensor along dimension 1, 2.
  RankedTensorType tensorType =
      RankedTensorType::get({3, 2, 4}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  LayoutAttr layout = LayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {1, 2};
  LayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 1);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  presburger::IntegerRelation expectedRelation = layout.getIntegerRelation();
  expectedRelation.projectOut(2, 1);
  expectedRelation.projectOut(1, 1);
  expectedRelation =
      LayoutAttr::getFromIntegerRelation(&context, expectedRelation)
          .getIntegerRelation();

  EXPECT_TRUE(isRelationEqual(reducedRelation, expectedRelation));
}

TEST(UtilsTest, TestFoldConvSpatialPadding) {
  MLIRContext context;
  Type elementType = IndexType::get(&context);

  // Rank 4: both spatial dims shrink by 2 * padding, (N, C) untouched.
  auto rank4 = RankedTensorType::get({1, 4, 6, 8}, elementType);
  auto folded4 = foldConvSpatialPadding(rank4, 1);
  ASSERT_TRUE(folded4.has_value());
  EXPECT_EQ(folded4->dataType,
            RankedTensorType::get({1, 4, 4, 6}, elementType));
  EXPECT_EQ(folded4->padding, 1);

  // Rank 3: only the width dim is spatial.
  auto rank3 = RankedTensorType::get({1, 4, 8}, elementType);
  auto folded3 = foldConvSpatialPadding(rank3, 2);
  ASSERT_TRUE(folded3.has_value());
  EXPECT_EQ(folded3->dataType, RankedTensorType::get({1, 4, 4}, elementType));

  // A padding of 0 is a no-op fold, not a rejection.
  auto unfolded = foldConvSpatialPadding(rank4, 0);
  ASSERT_TRUE(unfolded.has_value());
  EXPECT_EQ(unfolded->dataType, rank4);
}

TEST(UtilsTest, TestFoldConvSpatialPaddingRejectsBadPadding) {
  MLIRContext context;
  Type elementType = IndexType::get(&context);
  auto dataType = RankedTensorType::get({1, 4, 4, 4}, elementType);

  // A negative padding would grow the operand rather than shrink it, and every
  // resulting extent is still positive, so the emptiness check below does not
  // catch it.
  EXPECT_FALSE(foldConvSpatialPadding(dataType, -1).has_value());

  // A padding that consumes a whole spatial dim leaves nothing to convolve.
  EXPECT_FALSE(foldConvSpatialPadding(dataType, 2).has_value());

  // Ranks the conv kernels do not pack.
  EXPECT_FALSE(
      foldConvSpatialPadding(RankedTensorType::get({4, 4}, elementType), 1)
          .has_value());
}

TEST(UtilsTest, TestSetConvFoldedPaddingClearsStaleAttr) {
  MLIRContext context;
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> op = ModuleOp::create(builder.getUnknownLoc());

  EXPECT_EQ(getConvFoldedPadding(*op), 0);

  setConvFoldedPadding(*op, 2);
  EXPECT_EQ(getConvFoldedPadding(*op), 2);

  // A later run that folds nothing must clear the attribute rather than leave
  // the stale 2 behind for ConvertToCiphertextSemantics to act on.
  setConvFoldedPadding(*op, 0);
  EXPECT_FALSE((*op)->hasAttr(kConvFoldedPaddingAttrName));
  EXPECT_EQ(getConvFoldedPadding(*op), 0);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
