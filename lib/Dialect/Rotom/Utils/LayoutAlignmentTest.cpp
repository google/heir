#include <cstddef>
#include <cstdint>
#include <utility>

#include <string>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
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

TEST_F(LayoutAlignmentTest, DimAlignmentTracksCiphertextSide) {
  LayoutAttr lhs = layout({dim(0, 2), dim(1, 4)}, 4);
  LayoutAttr rhs = layout({dim(1, 2), dim(0, 4)}, 4);
  EXPECT_FALSE(rotom::layoutsAlignedByDimMap(lhs, rhs, {{0, 0}}));
  EXPECT_TRUE(rotom::layoutsAlignedByDimMap(lhs, rhs, {{0, 1}}));
}

// Reads a layout's rolls back as a list of (from, to) pairs.
static SmallVector<std::pair<int64_t, int64_t>> rollPairs(LayoutAttr layout) {
  SmallVector<std::pair<int64_t, int64_t>> pairs;
  DenseI64ArrayAttr rolls = layout.getRolls();
  if (!rolls) return pairs;
  ArrayRef<int64_t> flat = rolls.asArrayRef();
  for (size_t i = 0; i + 1 < flat.size(); i += 2) {
    pairs.push_back({flat[i], flat[i + 1]});
  }
  return pairs;
}

TEST_F(LayoutAlignmentTest, WithRollsReplacesExistingRolls) {
  LayoutAttr base = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rolled = rotom::withRolls(base, {{0, 1}});
  EXPECT_EQ(rollPairs(rolled),
            (SmallVector<std::pair<int64_t, int64_t>>{{0, 1}}));

  // A second call replaces, rather than appends to, the existing rolls.
  LayoutAttr rerolled = rotom::withRolls(rolled, {{1, 0}});
  EXPECT_EQ(rollPairs(rerolled),
            (SmallVector<std::pair<int64_t, int64_t>>{{1, 0}}));

  // dims and n are carried through unchanged.
  EXPECT_EQ(rerolled.getDims(), base.getDims());
  EXPECT_EQ(rerolled.getN(), base.getN());
}

TEST_F(LayoutAlignmentTest, EnumerateSingleRollsSquareLayout) {
  // A 4x4 row-major layout fully packed into one ciphertext: both dims are
  // slot-side and equal extent, so both roll orientations materialize.
  LayoutAttr base = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<LayoutAttr> variants = rotom::enumerateSingleRolls(base);

  ASSERT_EQ(variants.size(), 2);
  SmallVector<SmallVector<std::pair<int64_t, int64_t>>> seen;
  for (LayoutAttr variant : variants) {
    EXPECT_TRUE(rotom::isMaterializableRotomLayout(variant));
    EXPECT_EQ(variant.getDims(), base.getDims());
    seen.push_back(rollPairs(variant));
  }
  EXPECT_NE(seen[0], seen[1]);
  for (const auto& pairs : seen) {
    ASSERT_EQ(pairs.size(), 1);
    EXPECT_TRUE((pairs[0] == std::pair<int64_t, int64_t>{0, 1}) ||
                (pairs[0] == std::pair<int64_t, int64_t>{1, 0}));
  }
}

TEST_F(LayoutAlignmentTest, EnumerateSingleRollsRequiresEqualExtent) {
  // A 2x4 layout has no two traversal dims of equal extent, so no rolls.
  LayoutAttr base = layout({dim(0, 2), dim(1, 4)}, 8);
  EXPECT_TRUE(rotom::enumerateSingleRolls(base).empty());
}

TEST_F(LayoutAlignmentTest, EnumerateSingleRollsSkipsCiphertextSideDims) {
  // A 4x4 layout split across 4 ciphertexts: dim 0 is ciphertext-side, dim 1 is
  // slot-side. The dims share extent 4, but only dim 1 is slot-side, so there is
  // no slot-side pair to roll. (A roll referencing the ciphertext-side dim would
  // lower as a no-op, which is exactly what the slot-side restriction avoids.)
  LayoutAttr base = layout({dim(0, 4), dim(1, 4)}, 4);
  ASSERT_TRUE(rotom::isMaterializableRotomLayout(base));
  EXPECT_TRUE(rotom::enumerateSingleRolls(base).empty());
}

TEST_F(LayoutAlignmentTest, EnumerateSingleRollsChangesMaterializedLayout) {
  // Every enumerated variant must lower to a different ISL map than the base;
  // otherwise the roll is a no-op masquerading as a distinct candidate.
  LayoutAttr base = layout({dim(0, 4), dim(1, 4)}, 16);
  FailureOr<std::string> baseIsl =
      rotom::RotomTensorExtLayoutLowering::lowerToTensorExtIsl(base);
  ASSERT_TRUE(succeeded(baseIsl));

  SmallVector<LayoutAttr> variants = rotom::enumerateSingleRolls(base);
  ASSERT_FALSE(variants.empty());
  for (LayoutAttr variant : variants) {
    FailureOr<std::string> variantIsl =
        rotom::RotomTensorExtLayoutLowering::lowerToTensorExtIsl(variant);
    ASSERT_TRUE(succeeded(variantIsl));
    EXPECT_NE(*variantIsl, *baseIsl);
  }
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

}  // namespace
}  // namespace heir
}  // namespace mlir
