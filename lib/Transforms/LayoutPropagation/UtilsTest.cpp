#include <cstdint>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Transforms/LayoutPropagation/Utils.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project

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

TEST(UtilsTest, TestReduceNewLayout) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 4x6 tensor packed into a 3x8 tensor along dimension 0.
  RankedTensorType tensorType =
      RankedTensorType::get({4, 6}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  NewLayoutAttr layout =
      NewLayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {0};
  NewLayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 1);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  // The reduced layout should only have points from the original layout when
  // the reduced dimension is zero.
  for (int axes1 = 0; axes1 < 6; ++axes1) {
    for (int ct = 0; ct < 3; ++ct) {
      for (int slot = 0; slot < 8; ++slot) {
        bool atReducedDim =
            relation.containsPointNoLocal({0, axes1, ct, slot}).has_value();
        if (atReducedDim) {
          EXPECT_TRUE(reducedRelation.containsPointNoLocal({axes1, ct, slot})
                          .has_value());
        } else {
          // There also shouldn't be any other points.
          EXPECT_FALSE(reducedRelation.containsPointNoLocal({axes1, ct, slot})
                           .has_value());
        }
      }
    }
  }
}

TEST(UtilsTest, TestReduceNewLayoutMultiDim) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 3x2x4 tensor packed into a 3x8 tensor along dimension 2.
  RankedTensorType tensorType =
      RankedTensorType::get({3, 2, 4}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  NewLayoutAttr layout =
      NewLayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {2};
  NewLayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 2);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  // The reduced layout should only have points from the original layout when
  // the reduced dimension is zero.
  for (int axes0 = 0; axes0 < 3; ++axes0) {
    for (int axes1 = 0; axes1 < 2; ++axes1) {
      for (int ct = 0; ct < 3; ++ct) {
        for (int slot = 0; slot < 8; ++slot) {
          bool atReducedDim =
              relation.containsPointNoLocal({axes0, axes1, 0, ct, slot})
                  .has_value();
          if (atReducedDim) {
            EXPECT_TRUE(
                reducedRelation.containsPointNoLocal({axes0, axes1, ct, slot})
                    .has_value());
          } else {
            // There also shouldn't be any other points.
            EXPECT_FALSE(
                reducedRelation.containsPointNoLocal({axes0, axes1, ct, slot})
                    .has_value());
          }
        }
      }
    }
  }
}

TEST(UtilsTest, TestReduceNewLayoutManyReductions) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 3x2x4 tensor packed into a 3x8 tensor along dimension 1, 2.
  RankedTensorType tensorType =
      RankedTensorType::get({3, 2, 4}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  NewLayoutAttr layout =
      NewLayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {1, 2};
  NewLayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 1);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  // The reduced layout should only have points from the original layout when
  // the reduced dimensions are zero.
  for (int axes0 = 0; axes0 < 3; ++axes0) {
    for (int ct = 0; ct < 3; ++ct) {
      for (int slot = 0; slot < 8; ++slot) {
        bool atReducedDim =
            relation.containsPointNoLocal({axes0, 0, 0, ct, slot}).has_value();
        if (atReducedDim) {
          EXPECT_TRUE(reducedRelation.containsPointNoLocal({axes0, ct, slot})
                          .has_value());
        } else {
          // There also shouldn't be any other points.
          EXPECT_FALSE(reducedRelation.containsPointNoLocal({axes0, ct, slot})
                           .has_value());
        }
      }
    }
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
