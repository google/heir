#include <cstdint>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/ContractionAlignment.h"
#include "llvm/include/llvm/ADT/SmallVector.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::MatmulPlan;
using rotom::RotomDialect;

// Operand layouts are written in each tensor's own dims: for the lhs (i, k)
// dim 0 is i and dim 1 is k; for the rhs (k, j) dim 0 is k and dim 1 is j.
// Only computeLayout uses iteration-space dims (i=0, j=1, k=2).
class ContractionAlignmentTest : public ::testing::Test {
 protected:
  ContractionAlignmentTest() { context.loadDialect<RotomDialect>(); }

  DimAttr dim(int64_t dim, int64_t size, int64_t stride = 1) {
    return DimAttr::get(&context, dim, size, stride);
  }

  DimAttr repl(int64_t size) { return dim(/*dim=*/-1, size); }

  DimAttr gap(int64_t size) { return dim(/*dim=*/-2, size); }

  LayoutAttr layout(ArrayRef<Attribute> dims, int64_t n) {
    return LayoutAttr::get(&context, ArrayAttr::get(&context, dims), n);
  }

  LayoutAttr rolledLayout(ArrayRef<Attribute> dims, int64_t n,
                          ArrayRef<int64_t> rolls) {
    return LayoutAttr::get(&context, ArrayAttr::get(&context, dims), n,
                           DenseI64ArrayAttr::get(&context, rolls));
  }

  const MatmulPlan* findPlan(ArrayRef<MatmulPlan> plans,
                             LayoutAttr computeLayout) {
    for (const MatmulPlan& plan : plans) {
      if (plan.computeLayout == computeLayout) return &plan;
    }
    return nullptr;
  }

  MLIRContext context;
};

// 4x4 matmul at n=16, both operands row-major. The four host x region
// variants dedup to three roll-free footprints (rhs-hosted/ct coincides with
// lhs-hosted/slot); each footprint adds two rolled diagonal variants --
// [2:4][1:4][0:4] rolls its ciphertext k by j or i (ct-diagonal), and
// [0:4][2:4][1:4] / [1:4][0:4][2:4] roll their slot k by the ciphertext
// piece or the other slot piece (slot-diagonal) -- for nine plans total.
TEST_F(ContractionAlignmentTest, RowMajorPairEnumeratesNineDedupedPlans) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);
  EXPECT_EQ(plans.size(), 9u);
}

TEST_F(ContractionAlignmentTest, LhsHostedSlotPlan) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  // j appended innermost: i indexes 4 ciphertexts, (k, j) fill the slots.
  const MatmulPlan* plan =
      findPlan(plans, layout({dim(0, 4), dim(2, 4), dim(1, 4)}, 16));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs, layout({dim(0, 4), dim(1, 4), repl(4)}, 16));
  EXPECT_EQ(plan->expandedRhs, layout({repl(4), dim(0, 4), dim(1, 4)}, 16));
  EXPECT_EQ(plan->resultLayout, layout({dim(0, 4), gap(4), dim(1, 4)}, 16));
  // lhs fills j in slots: log2(4) rotate-and-adds on each of 4 ciphertexts.
  EXPECT_EQ(plan->lhsFillRotations, 8);
  EXPECT_EQ(plan->lhsFillAdds, 8);
  // rhs replicates across ciphertexts: free.
  EXPECT_EQ(plan->rhsFillRotations, 0);
  EXPECT_EQ(plan->rhsFillAdds, 0);
  // k lives in slots: log2(4) rotate-and-reduce on each of 4 ciphertexts
  // (only the k=0 offset holds the true sum, hence the gap in the result).
  EXPECT_EQ(plan->reduceRotations, 8);
  EXPECT_EQ(plan->reduceAdds, 8);
}

TEST_F(ContractionAlignmentTest, LhsHostedCtPlan) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  // j prepended outermost: j indexes 4 ciphertexts, (i, k) fill the slots.
  const MatmulPlan* plan =
      findPlan(plans, layout({dim(1, 4), dim(0, 4), dim(2, 4)}, 16));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs, layout({repl(4), dim(0, 4), dim(1, 4)}, 16));
  EXPECT_EQ(plan->expandedRhs, layout({dim(1, 4), repl(4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->resultLayout, layout({dim(1, 4), dim(0, 4), gap(4)}, 16));
  // lhs replicates across ciphertexts (free); rhs fills i in slots.
  EXPECT_EQ(plan->lhsFillRotations, 0);
  EXPECT_EQ(plan->lhsFillAdds, 0);
  EXPECT_EQ(plan->rhsFillRotations, 8);
  EXPECT_EQ(plan->rhsFillAdds, 8);
  EXPECT_EQ(plan->reduceRotations, 8);
  EXPECT_EQ(plan->reduceAdds, 8);
}

TEST_F(ContractionAlignmentTest, RhsHostedSlotPlanSumsAcrossCiphertexts) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  // i appended innermost to the rhs host: k indexes 4 ciphertexts, so the
  // reduction is pure ciphertext adds and the result collapses to one
  // ciphertext with no replication left over.
  const MatmulPlan* plan =
      findPlan(plans, layout({dim(2, 4), dim(1, 4), dim(0, 4)}, 16));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs, layout({dim(1, 4), repl(4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->expandedRhs, layout({dim(0, 4), dim(1, 4), repl(4)}, 16));
  EXPECT_EQ(plan->resultLayout, layout({dim(1, 4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->lhsFillRotations, 8);
  EXPECT_EQ(plan->rhsFillRotations, 8);
  EXPECT_EQ(plan->reduceRotations, 0);
  EXPECT_EQ(plan->reduceAdds, 3);
}

// A free dim of extent 1 (matvec-shaped: rhs has no j pieces) needs no
// insertion, and the rhs-hosted ct variant coincides with the lhs host.
// Both single-ciphertext footprints add a slot-diagonal rolled variant
// (roll the slot k by the slot i) -- the Halevi-Shoup diagonal matvec.
TEST_F(ContractionAlignmentTest, UnitFreeDimSkipsInsertion) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);
  EXPECT_EQ(plans.size(), 4u);

  const MatmulPlan* plan = findPlan(plans, layout({dim(0, 4), dim(2, 4)}, 16));
  ASSERT_NE(plan, nullptr);
  // No j to replicate: the lhs is already at the compute placement.
  EXPECT_EQ(plan->expandedLhs, lhs);
  EXPECT_EQ(plan->expandedRhs, layout({repl(4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->resultLayout, layout({dim(0, 4), gap(4)}, 16));
  EXPECT_EQ(plan->lhsFillRotations, 0);
  // One ciphertext: replicating i over 4 slot positions is log2(4) = 2.
  EXPECT_EQ(plan->rhsFillRotations, 2);
  EXPECT_EQ(plan->reduceRotations, 2);
  EXPECT_EQ(plan->reduceAdds, 2);
}

// The ct/slot split is inferred greedily from the right, so a
// non-power-of-two free extent never lands in the slot region -- every
// placement pushes it (and the pieces outside the surviving slot suffix)
// into the ciphertext prefix, where replication is free. Three roll-free
// footprints plus two rolled variants: the ciphertext k rolled by its
// same-extent slot i piece, and a slot k rolled by the slot i piece (the
// non-power-of-two j piece is never an eligible partner).
TEST_F(ContractionAlignmentTest, NonPowerOfTwoFreeExtentStaysInCtRegion) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 3)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);
  EXPECT_EQ(plans.size(), 5u);
  for (const MatmulPlan& plan : plans) {
    EXPECT_EQ(plan.lhsFillRotations, 0);
    EXPECT_EQ(plan.lhsFillAdds, 0);
  }
}

// A mixed-radix split axis in the host layout carries through the compute
// placement, and a k piece in the ciphertext prefix is summed by ciphertext
// adds and dropped from the result.
TEST_F(ContractionAlignmentTest, MixedRadixHostCarriesThroughAndDropsCtK) {
  LayoutAttr lhs =
      layout({dim(0, 2, /*stride=*/2), dim(1, 4), dim(0, 2, /*stride=*/1)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  const MatmulPlan* plan =
      findPlan(plans, layout({dim(0, 2, /*stride=*/2), dim(2, 4),
                              dim(0, 2, /*stride=*/1), dim(1, 4)},
                             16));
  ASSERT_NE(plan, nullptr);
  // Compute: ct prefix [0:2:2][2:4] (8 ciphertexts), slots [0:2:1][1:4].
  // Summing k drops its ct piece; the split i axis survives intact.
  EXPECT_EQ(plan->resultLayout, layout({dim(0, 2, /*stride=*/2),
                                        dim(0, 2, /*stride=*/1), dim(1, 4)},
                                       16));
  EXPECT_EQ(plan->reduceRotations, 0);
  // 8 ciphertexts collapse to 2: six ciphertext adds.
  EXPECT_EQ(plan->reduceAdds, 6);
}

// The rolled ct-diagonal sibling of RhsHostedSlotPlanSumsAcrossCiphertexts:
// same footprint [2:4][1:4][0:4], with the ciphertext k piece rolled by the
// slot i piece (positions 0 and 2). Ciphertext c of the product holds
// A[x, (c+x)%4] * B[(c+x)%4, y] at slot (y, x), so the plain ciphertext adds
// still sum k and the result is one unrolled column-major ciphertext. Both
// expansions inherit the positional roll: the lhs owns i (roll by a
// traversal dim); the rhs sees the replication that subsumed i (roll by
// replication -- every rotation of k materialized across its blocks).
TEST_F(ContractionAlignmentTest, RolledCtDiagonalPlanKeepsFootprintCounts) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  const MatmulPlan* plan = findPlan(
      plans,
      rolledLayout({dim(2, 4), dim(1, 4), dim(0, 4)}, 16, /*rolls=*/{0, 2}));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs,
            rolledLayout({dim(1, 4), repl(4), dim(0, 4)}, 16, {0, 2}));
  EXPECT_EQ(plan->expandedRhs,
            rolledLayout({dim(0, 4), dim(1, 4), repl(4)}, 16, {0, 2}));
  EXPECT_EQ(plan->resultLayout, layout({dim(1, 4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->reduceRotations, 0);
  EXPECT_EQ(plan->reduceAdds, 3);
}

// Operands seeded directly at a rolled expanded placement (the ct-diagonal
// pair a = roll(0,2) [1:4];[0:4][R:4], b = roll(0,2) [0:4];[R:4][1:4]):
// replacing the lhs host's replication piece with j recovers the compute
// footprint [2:4];[0:4][1:4], and its (0,2)-rolled variant expands each
// operand back to exactly its input layout -- a zero-conversion plan whose
// kernel is 4 ciphertext multiplies and 3 adds with no rotations, leaving
// one row-major result ciphertext.
TEST_F(ContractionAlignmentTest, ExpandedRolledPairYieldsZeroConversionPlan) {
  LayoutAttr lhs =
      rolledLayout({dim(1, 4), dim(0, 4), repl(4)}, 16, /*rolls=*/{0, 2});
  LayoutAttr rhs =
      rolledLayout({dim(0, 4), repl(4), dim(1, 4)}, 16, /*rolls=*/{0, 2});
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  const MatmulPlan* plan = findPlan(
      plans,
      rolledLayout({dim(2, 4), dim(0, 4), dim(1, 4)}, 16, /*rolls=*/{0, 2}));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs, lhs);
  EXPECT_EQ(plan->expandedRhs, rhs);
  EXPECT_EQ(plan->resultLayout, layout({dim(0, 4), dim(1, 4)}, 16));
  EXPECT_EQ(plan->reduceRotations, 0);
  EXPECT_EQ(plan->reduceAdds, 3);
}

// The slot-diagonal (Halevi-Shoup) family: hosting the column-major lhs
// ([k][i] slots) and prepending j gives the footprint [j:ct];[k][i], whose
// slot k rolls by the slot i piece -- ciphertext j, slot (k', x) computes
// A[x, (k'+x)%4] * B[(k'+x)%4, j], the classic diagonal kernel. The lhs
// expansion is the diagonalized matrix replicated across ciphertexts
// (reachable from a one-ciphertext diagonal packing by free replication);
// summing k is the usual slot rotate-and-reduce and gaps the rolled piece.
TEST_F(ContractionAlignmentTest, SlotDiagonalPlanRollsSlotKBySlotPartner) {
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  const MatmulPlan* plan = findPlan(
      plans,
      rolledLayout({dim(1, 4), dim(2, 4), dim(0, 4)}, 16, /*rolls=*/{1, 2}));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs,
            rolledLayout({repl(4), dim(1, 4), dim(0, 4)}, 16, {1, 2}));
  EXPECT_EQ(plan->expandedRhs,
            rolledLayout({dim(1, 4), dim(0, 4), repl(4)}, 16, {1, 2}));
  EXPECT_EQ(plan->resultLayout, layout({dim(1, 4), gap(4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->reduceRotations, 8);
  EXPECT_EQ(plan->reduceAdds, 8);
}

// The same footprint rolled by its ciphertext piece instead: the lhs
// expansion becomes the replicate-then-roll placement (its rolled slot k is
// outermost, so expanding a compact column-major source is one rotation per
// ciphertext).
TEST_F(ContractionAlignmentTest, SlotDiagonalPlanRollsSlotKByCtPartner) {
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  const MatmulPlan* plan = findPlan(
      plans,
      rolledLayout({dim(1, 4), dim(2, 4), dim(0, 4)}, 16, /*rolls=*/{1, 0}));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs,
            rolledLayout({repl(4), dim(1, 4), dim(0, 4)}, 16, {1, 0}));
  EXPECT_EQ(plan->expandedRhs,
            rolledLayout({dim(1, 4), dim(0, 4), repl(4)}, 16, {1, 0}));
  EXPECT_EQ(plan->resultLayout, layout({dim(1, 4), gap(4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->reduceRotations, 8);
  EXPECT_EQ(plan->reduceAdds, 8);
}

// The reverse-subsumption host variant also serves roll-free layouts: an
// operand already at an expanded placement (replication piece included)
// enumerates the compute placement it came from, so realigning it is free.
TEST_F(ContractionAlignmentTest, ExpandedRollFreeHostReusesReplicationPiece) {
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4), repl(4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), repl(4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);

  const MatmulPlan* plan =
      findPlan(plans, layout({dim(2, 4), dim(0, 4), dim(1, 4)}, 16));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->expandedLhs, lhs);
  EXPECT_EQ(plan->expandedRhs, rhs);
  EXPECT_EQ(plan->resultLayout, layout({dim(0, 4), dim(1, 4)}, 16));
}

TEST_F(ContractionAlignmentTest, MismatchedNReturnsNoPlans) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 32);
  EXPECT_TRUE(rotom::enumerateMatmulPlans(lhs, rhs).empty());
}

}  // namespace
}  // namespace heir
}  // namespace mlir
