#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::RotomDialect;

class LayoutAlignmentTest : public ::testing::Test {
 protected:
  LayoutAlignmentTest() { context.loadDialect<RotomDialect>(); }

  DimAttr dim(int64_t dim, int64_t size, int64_t stride = 1) {
    return DimAttr::get(&context, dim, size, stride);
  }

  LayoutAttr layout(ArrayRef<Attribute> dims, int64_t n) {
    return LayoutAttr::get(&context, ArrayAttr::get(&context, dims), n);
  }

  MLIRContext context;
};

TEST_F(LayoutAlignmentTest, UnitStridedTraversalRuleAllowsReplicateStride) {
  LayoutAttr replicated =
      layout({dim(0, 4), dim(/*dim=*/-1, /*size=*/2, /*stride=*/4)}, 8);
  EXPECT_TRUE(rotom::hasOnlyUnitStridedTraversalDims(replicated));
  EXPECT_TRUE(rotom::supportsRotomAlignmentLowering(replicated, replicated,
                                                    replicated));
}

TEST_F(LayoutAlignmentTest, UnitStridedTraversalRuleRejectsTraversalStride) {
  LayoutAttr rolled =
      layout({dim(0, 2, /*stride=*/2), dim(0, 2), dim(1, 4)}, 8);
  EXPECT_FALSE(rotom::hasOnlyUnitStridedTraversalDims(rolled));
  EXPECT_FALSE(rotom::supportsRotomAlignmentLowering(rolled, rolled, rolled));
}

TEST_F(LayoutAlignmentTest, RejectsDifferentCiphertextSizes) {
  LayoutAttr n8 = layout({dim(0, 2), dim(1, 4)}, 8);
  LayoutAttr n16 = layout({dim(0, 2), dim(1, 4)}, 16);
  EXPECT_FALSE(rotom::supportsRotomAlignmentLowering(n8, n16, n8));
}

TEST_F(LayoutAlignmentTest, RejectsMismatchedCiphertextCounts) {
  LayoutAttr oneCiphertext = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr fourCiphertexts = layout({dim(0, 4), dim(1, 4)}, 4);
  EXPECT_EQ(rotom::layoutNumCiphertexts(oneCiphertext), 1);
  EXPECT_EQ(rotom::layoutNumCiphertexts(fourCiphertexts), 4);
  EXPECT_FALSE(rotom::supportsRotomAlignmentLowering(
      oneCiphertext, oneCiphertext, fourCiphertexts));
}

TEST_F(LayoutAlignmentTest, RejectsNonMaterializableRepeatedDimLayout) {
  LayoutAttr repeated =
      layout({dim(0, 2), dim(1, 2), dim(0, 2), dim(1, 2)}, /*n=*/16);
  EXPECT_TRUE(rotom::hasOnlyUnitStridedTraversalDims(repeated));
  EXPECT_FALSE(rotom::isMaterializableRotomLayout(repeated));
  EXPECT_FALSE(
      rotom::supportsRotomAlignmentLowering(repeated, repeated, repeated));
}

TEST_F(LayoutAlignmentTest, DimAlignmentTracksCiphertextSide) {
  LayoutAttr lhs = layout({dim(0, 2), dim(1, 4)}, 4);
  LayoutAttr rhs = layout({dim(1, 2), dim(0, 4)}, 4);
  EXPECT_FALSE(rotom::layoutsAlignedByDimMap(lhs, rhs, {{0, 0}}));
  EXPECT_TRUE(rotom::layoutsAlignedByDimMap(lhs, rhs, {{0, 1}}));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
