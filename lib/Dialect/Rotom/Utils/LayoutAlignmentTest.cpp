#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include <string>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::RotomDialect;

class LayoutAlignmentTest : public ::testing::Test {
 protected:
  LayoutAlignmentTest() {
    context.loadDialect<RotomDialect>();
    context.loadDialect<tensor_ext::TensorExtDialect>();
  }

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

TEST_F(LayoutAlignmentTest, CountsSplitAxisByCiphertextPart) {
  // A 4x8 layout at n=16 whose row axis (extent 4) is split explicitly into a
  // high (ciphertext) piece [stride 2] and a low (slot) piece [stride 1] -- the
  // way to express an axis spanning the ct/slot boundary now that there is no
  // straddle auto-split. The column axis (extent 8) and the row's low piece
  // (extent 2) fill the 16 slots (8 * 2 == 16); the row's high piece (extent 2)
  // indexes ciphertexts. The layout therefore occupies 2 ciphertexts, not 4
  // (the full row extent).
  LayoutAttr split = layout({dim(0, 2, /*stride=*/2), dim(1, 8, /*stride=*/1),
                             dim(0, 2, /*stride=*/1)},
                            /*n=*/16);
  EXPECT_EQ(rotom::layoutNumCiphertexts(split), 2);
}

TEST_F(LayoutAlignmentTest, MaterializesMixedRadixRepeatedDimLayout) {
  // A repeated dim id is a mixed-radix split of one tensor axis: the two pieces
  // of axis 0 (strides 2 and 1) and of axis 1 share their axis's domain
  // variable. This is a valid, materializable 2x2-tiled layout. (An invalid
  // split -- e.g. two stride-1 pieces -- is rejected by LayoutAttr::verify.)
  LayoutAttr split = layout({dim(0, 2, /*stride=*/2), dim(1, 2, /*stride=*/2),
                             dim(0, 2, /*stride=*/1), dim(1, 2, /*stride=*/1)},
                            /*n=*/8);
  EXPECT_TRUE(rotom::isMaterializableRotomLayout(split));
}

TEST_F(LayoutAlignmentTest, ConversionMovesIdenticalIsEmpty) {
  LayoutAttr a = layout({dim(0, 4)}, 4);
  EXPECT_TRUE(rotom::conversionMoves(a, a).empty());
}

TEST_F(LayoutAlignmentTest, ConversionMovesSplitEquivalentIsEmpty) {
  // [0:4:1] packs axis 0 identically to the mixed-radix split [0:2:2][0:2:1],
  // so aligning the two needs no rotations.
  LayoutAttr whole = layout({dim(0, 4)}, 4);
  LayoutAttr split =
      layout({dim(0, 2, /*stride=*/2), dim(0, 2, /*stride=*/1)}, 4);
  EXPECT_TRUE(rotom::conversionMoves(whole, split).empty());
  EXPECT_TRUE(rotom::conversionMoves(split, whole).empty());
}

TEST_F(LayoutAlignmentTest, ConversionMovesSwappedSlotsReportsMoves) {
  // Row-major vs column-major 2x2: the two slot bits swap positions.
  LayoutAttr rowMajor = layout({dim(0, 2), dim(1, 2)}, 4);
  LayoutAttr colMajor = layout({dim(1, 2), dim(0, 2)}, 4);
  EXPECT_EQ(rotom::conversionMoves(rowMajor, colMajor).size(), 2u);
}

TEST_F(LayoutAlignmentTest, ConversionMovesIgnoresCiphertextOrder) {
  // Same slot dim (axis 2); the two ciphertext-side axes are merely ordered
  // differently, which is a free ciphertext relabel -- no slot moves.
  LayoutAttr a = layout({dim(0, 2), dim(1, 2), dim(2, 2)}, 2);
  LayoutAttr b = layout({dim(1, 2), dim(0, 2), dim(2, 2)}, 2);
  EXPECT_TRUE(rotom::conversionMoves(a, b).empty());
}

TEST_F(LayoutAlignmentTest, ShiftNetworkConversionCostIsZeroForEqual) {
  LayoutAttr a = layout({dim(0, 2), dim(1, 2)}, 4);
  std::optional<int64_t> cost = rotom::shiftNetworkConversionCost(a, a);
  ASSERT_TRUE(cost.has_value());
  EXPECT_EQ(*cost, 0);
}

TEST_F(LayoutAlignmentTest, ShiftNetworkConversionCostHasValueForSwap) {
  // Row-major vs column-major 2x2 is a genuine slot permutation, so the shift
  // network reports a (non-negative) rotation cost via the tensor_ext bridge.
  LayoutAttr rowMajor = layout({dim(0, 2), dim(1, 2)}, 4);
  LayoutAttr colMajor = layout({dim(1, 2), dim(0, 2)}, 4);
  std::optional<int64_t> cost =
      rotom::shiftNetworkConversionCost(rowMajor, colMajor);
  ASSERT_TRUE(cost.has_value());
  EXPECT_GE(*cost, 0);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
