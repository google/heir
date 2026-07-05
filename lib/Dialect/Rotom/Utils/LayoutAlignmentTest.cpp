#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

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

TEST_F(LayoutAlignmentTest, ExpansionPureCtReplicationIsFreeCopies) {
  // 4x4 row-major (1 ct at n=16) expanded to 4 replicated ciphertexts:
  // every step is a full-row, zero-shift copy -- no rotations, no masks.
  LayoutAttr source = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr expanded = layout({dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)}, 16);
  auto steps = rotom::planLayoutExpansion(source, expanded);
  ASSERT_TRUE(succeeded(steps));
  ASSERT_EQ(steps->size(), 4u);
  for (const rotom::LayoutExpansionStep& step : *steps) {
    EXPECT_EQ(step.sourceCt, 0);
    EXPECT_EQ(step.shift, 0);
    EXPECT_EQ(step.targetSlots.size(), 16u);
  }
}

TEST_F(LayoutAlignmentTest, ExpansionScatterNeedsRotationsAndMasks) {
  // Expanding 4x4 row-major so that ciphertext i holds row i replicated
  // ([0:4] to ct, [1:4] to slots, replication innermost): each target
  // ciphertext draws 4 slot-groups from the single source ciphertext, so
  // steps carry nonzero shifts and partial-row masks.
  LayoutAttr source = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr expanded = layout({dim(0, 4), dim(1, 4), dim(/*dim=*/-1, 4)}, 16);
  auto steps = rotom::planLayoutExpansion(source, expanded);
  ASSERT_TRUE(succeeded(steps));
  EXPECT_GT(steps->size(), 4u);
  bool sawShift = false;
  bool sawMask = false;
  for (const rotom::LayoutExpansionStep& step : *steps) {
    if (step.shift != 0) sawShift = true;
    if (step.targetSlots.size() != 16u) sawMask = true;
  }
  EXPECT_TRUE(sawShift);
  EXPECT_TRUE(sawMask);
}

TEST_F(LayoutAlignmentTest, CompactionOfGappedMatmulResultPlansMaskedGathers) {
  // Compaction (the ciphertext-count-DECREASING direction): a matmul result
  // layout [0:4:1];[1:4:1][G:4:1] -- 4 ciphertexts, row i claimed at the k=0
  // offsets slot 4j, everything else gap garbage -- compacts into one
  // column-major ciphertext (slot = 4j + i). Each source ciphertext i
  // contributes one group: shift (-i) mod 16, target slots {4j + i}, masked
  // (the mask also kills the gap garbage).
  LayoutAttr source = layout({dim(0, 4), dim(1, 4), dim(/*dim=*/-2, 4)}, 16);
  LayoutAttr compact = layout({dim(1, 4), dim(0, 4)}, 16);
  ASSERT_EQ(rotom::layoutNumCiphertexts(source), 4);
  ASSERT_EQ(rotom::layoutNumCiphertexts(compact), 1);

  auto steps = rotom::planLayoutExpansion(source, compact);
  ASSERT_TRUE(succeeded(steps));
  ASSERT_EQ(steps->size(), 4u);
  for (const rotom::LayoutExpansionStep& step : *steps) {
    EXPECT_EQ(step.targetCt, 0);
    const int64_t i = step.sourceCt;
    EXPECT_EQ(step.shift, (16 - i) % 16);
    ASSERT_EQ(step.targetSlots.size(), 4u);
    for (int64_t j = 0; j < 4; ++j) {
      EXPECT_EQ(step.targetSlots[j], 4 * j + i);
    }
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
