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

  LayoutAttr layout(ArrayRef<Attribute> dims, int64_t n) {
    return LayoutAttr::get(&context, ArrayAttr::get(&context, dims), n);
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
// variants dedup to three plans: rhs-hosted/ct coincides with
// lhs-hosted/slot.
TEST_F(ContractionAlignmentTest, RowMajorPairEnumeratesThreeDedupedPlans) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);
  EXPECT_EQ(plans.size(), 3u);
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
  EXPECT_EQ(plan->resultLayout, layout({dim(0, 4), repl(4), dim(1, 4)}, 16));
  // lhs fills j in slots: log2(4) rotate-and-adds on each of 4 ciphertexts.
  EXPECT_EQ(plan->lhsFillRotations, 8);
  EXPECT_EQ(plan->lhsFillAdds, 8);
  // rhs replicates across ciphertexts: free.
  EXPECT_EQ(plan->rhsFillRotations, 0);
  EXPECT_EQ(plan->rhsFillAdds, 0);
  // k lives in slots: log2(4) rotate-and-reduce on each of 4 ciphertexts.
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
  EXPECT_EQ(plan->resultLayout, layout({dim(1, 4), dim(0, 4), repl(4)}, 16));
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
TEST_F(ContractionAlignmentTest, UnitFreeDimSkipsInsertion) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);
  EXPECT_EQ(plans.size(), 2u);

  const MatmulPlan* plan = findPlan(plans, layout({dim(0, 4), dim(2, 4)}, 16));
  ASSERT_NE(plan, nullptr);
  // No j to replicate: the lhs is already at the compute placement.
  EXPECT_EQ(plan->expandedLhs, lhs);
  EXPECT_EQ(plan->expandedRhs, layout({repl(4), dim(0, 4)}, 16));
  EXPECT_EQ(plan->resultLayout, layout({dim(0, 4), repl(4)}, 16));
  EXPECT_EQ(plan->lhsFillRotations, 0);
  // One ciphertext: replicating i over 4 slot positions is log2(4) = 2.
  EXPECT_EQ(plan->rhsFillRotations, 2);
  EXPECT_EQ(plan->reduceRotations, 2);
  EXPECT_EQ(plan->reduceAdds, 2);
}

// The ct/slot split is inferred greedily from the right, so a
// non-power-of-two free extent never lands in the slot region -- every
// placement pushes it (and the pieces outside the surviving slot suffix)
// into the ciphertext prefix, where replication is free.
TEST_F(ContractionAlignmentTest, NonPowerOfTwoFreeExtentStaysInCtRegion) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 3)}, 16);
  SmallVector<MatmulPlan> plans = rotom::enumerateMatmulPlans(lhs, rhs);
  EXPECT_EQ(plans.size(), 3u);
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
  // The k ciphertext piece is not outermost in the prefix ([0:2:2] is), so
  // the v1 row-slice reduction cannot realize this plan.
  EXPECT_FALSE(rotom::isLowerableMatmulPlan(*plan, lhs, rhs));
}

TEST_F(ContractionAlignmentTest, MismatchedNReturnsNoPlans) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 32);
  EXPECT_TRUE(rotom::enumerateMatmulPlans(lhs, rhs).empty());
}

TEST_F(ContractionAlignmentTest, StripOuterCtReplication) {
  // [-1:4] indexes 4 replicated ciphertexts; stripping it leaves the
  // single-ciphertext inner layout.
  LayoutAttr expanded = layout({repl(4), dim(0, 4), dim(1, 4)}, 16);
  int64_t factor = 0;
  LayoutAttr inner = rotom::stripOuterCtReplication(expanded, factor);
  EXPECT_EQ(factor, 4);
  EXPECT_EQ(inner, layout({dim(0, 4), dim(1, 4)}, 16));

  // No ciphertext-region replication: unchanged.
  LayoutAttr slotOnly = layout({dim(0, 4), dim(1, 4), repl(4)}, 64);
  factor = 0;
  EXPECT_EQ(rotom::stripOuterCtReplication(slotOnly, factor), slotOnly);
  EXPECT_EQ(factor, 1);
}

// At n=16 the 4x4 iteration space (64 elements) cannot stay within the
// operands' single ciphertexts: a plan is v1-lowerable only when each
// expanded placement is outermost ciphertext copies over a layout with the
// operand's own ciphertext count.
TEST_F(ContractionAlignmentTest, LowerabilityRequiresMatchingCtCounts) {
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(1, 4)}, 16);

  // lhs-hosted/ct plan: expandedLhs is pure ciphertext copies (fine), but
  // expandedRhs spreads j across 4 ciphertexts from a 1-ciphertext source.
  SmallVector<MatmulPlan> plans16 = rotom::enumerateMatmulPlans(lhs, rhs);
  const MatmulPlan* p16 =
      findPlan(plans16, layout({dim(1, 4), dim(0, 4), dim(2, 4)}, 16));
  ASSERT_NE(p16, nullptr);
  EXPECT_FALSE(rotom::isLowerableMatmulPlan(*p16, lhs, rhs));

  // At n=64 the whole iteration space fits one ciphertext's slots, so every
  // plan is a same-shape conversion: lowerable.
  LayoutAttr lhs64 = layout({dim(0, 4), dim(1, 4)}, 64);
  LayoutAttr rhs64 = layout({dim(0, 4), dim(1, 4)}, 64);
  SmallVector<MatmulPlan> plans64 = rotom::enumerateMatmulPlans(lhs64, rhs64);
  ASSERT_FALSE(plans64.empty());
  for (const MatmulPlan& plan : plans64) {
    EXPECT_TRUE(rotom::isLowerableMatmulPlan(plan, lhs64, rhs64));
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
