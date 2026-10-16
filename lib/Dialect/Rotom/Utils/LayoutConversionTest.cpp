#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/LayoutConversion.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project  // from @googletest
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::RotomDialect;

class LayoutConversionTest : public ::testing::Test {
 protected:
  LayoutConversionTest() {
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

// The estimate is the "is this conversion free?" test: zero for every
// form of the same packing, in both directions, and nonzero otherwise.
TEST_F(LayoutConversionTest, EstimateIsZeroForIdenticalLayouts) {
  LayoutAttr a = layout({dim(0, 4)}, 4);
  EXPECT_EQ(rotom::estimateConversionCost(a, a).rotations, 0);
}

TEST_F(LayoutConversionTest, EstimateIsZeroForSplitEquivalentBothWays) {
  // [0:4:1] packs axis 0 identically to the split [0:2:2][0:2:1].
  LayoutAttr whole = layout({dim(0, 4)}, 4);
  LayoutAttr split =
      layout({dim(0, 2, /*stride=*/2), dim(0, 2, /*stride=*/1)}, 4);
  EXPECT_EQ(rotom::estimateConversionCost(whole, split).rotations, 0);
  EXPECT_EQ(rotom::estimateConversionCost(split, whole).rotations, 0);
}

TEST_F(LayoutConversionTest, EstimateIsZeroAcrossRollFormBothWays) {
  LayoutAttr whole = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, 4), dim(1, 4)}), /*n=*/16,
      DenseI64ArrayAttr::get(&context, {1, 0}));
  LayoutAttr split = LayoutAttr::get(
      &context,
      ArrayAttr::get(&context, {dim(0, 4), dim(1, 2, /*stride=*/2), dim(1, 2)}),
      /*n=*/16,
      DenseI64ArrayAttr::get(&context,
                             {rotom::encodeRollArg({/*isAxis=*/true, 1}), 0}));
  EXPECT_EQ(rotom::estimateConversionCost(whole, split).rotations, 0);
  EXPECT_EQ(rotom::estimateConversionCost(split, whole).rotations, 0);
}

TEST_F(LayoutConversionTest, EstimateIsZeroForSharedGap) {
  LayoutAttr gapped = layout({dim(0, 4), dim(/*dim=*/-2, 2)}, 8);
  EXPECT_EQ(rotom::estimateConversionCost(gapped, gapped).rotations, 0);
}

TEST_F(LayoutConversionTest, EstimateIgnoresCiphertextOrder) {
  // The two ciphertext-side axes are merely ordered differently: a free
  // ciphertext relabel.
  LayoutAttr a = layout({dim(0, 2), dim(1, 2), dim(2, 2)}, 2);
  LayoutAttr b = layout({dim(1, 2), dim(0, 2), dim(2, 2)}, 2);
  EXPECT_EQ(rotom::estimateConversionCost(a, b).rotations, 0);
}

TEST_F(LayoutConversionTest, EstimateIsNonzeroForSwappedSlots) {
  LayoutAttr rowMajor = layout({dim(0, 2), dim(1, 2)}, 4);
  LayoutAttr colMajor = layout({dim(1, 2), dim(0, 2)}, 4);
  EXPECT_GT(rotom::estimateConversionCost(rowMajor, colMajor).rotations, 0);
}

TEST_F(LayoutConversionTest, EstimateIsNonzeroForMaterializedRotations) {
  LayoutAttr plain = layout({dim(/*dim=*/-1, 4), dim(0, 4)}, 16);
  LayoutAttr rotations = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(/*dim=*/-1, 4), dim(0, 4)}),
      /*n=*/16, DenseI64ArrayAttr::get(&context, {1, 0}));
  EXPECT_GT(rotom::estimateConversionCost(plain, rotations).rotations, 0);
}

TEST_F(LayoutConversionTest, EstimateIsNonzeroWhenRollsDiffer) {
  LayoutAttr plain = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rolled = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, 4), dim(1, 4)}), /*n=*/16,
      DenseI64ArrayAttr::get(&context, {1, 0}));
  EXPECT_GT(rotom::estimateConversionCost(plain, rolled).rotations, 0);
}

// A new gap is free only when it leaves every other piece at its address.
// A trailing slot gap moves piece 0 from offset 1 to offset 2 (slot i ->
// slot 2i). That is not rotations alone: the plan rotates 3 times, masks
// each of the 4 copies (the unrotated one too) down to one slot, and adds
// them -- 3 rotations, 4 masks, 3 adds, which the estimate reproduces
// exactly. A leading slot gap leaves piece 0 at offset 1, so it is free.
TEST_F(LayoutConversionTest, NewGapCostsWhenItShiftsOtherPieces) {
  LayoutAttr plain = layout({dim(0, 4)}, 8);
  LayoutAttr gapLast = layout({dim(0, 4), dim(/*dim=*/-2, 2)}, 8);
  LayoutAttr gapFirst = layout({dim(/*dim=*/-2, 2), dim(0, 4)}, 8);

  auto est = rotom::estimateConversionCost(plain, gapLast);
  EXPECT_EQ(est.rotations, 3);
  EXPECT_EQ(est.masks, 4);  // the unrotated copy is masked too
  EXPECT_EQ(est.accumulates, 3);

  auto plan = rotom::planLayoutConversion(plain, gapLast);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->steps.size(), 4u);
  int rotations = 0;
  for (const auto& step : plan->steps) {
    if (step.shift != 0) ++rotations;
    EXPECT_EQ(step.targetSlots.size(), 1u);  // every copy is masked
  }
  EXPECT_EQ(rotations, 3);

  EXPECT_EQ(rotom::estimateConversionCost(plain, gapFirst).rotations, 0);
}

// A conversion moves addresses, not content. Two layouts with the same
// piece order but different rolls hold different values at the same
// addresses, so no conversion relates them; the planner must refuse rather
// than return an identity plan. A roll names its arguments by piece position,
// so the same relation written against a different piece order is a different
// roll and is refused too.
TEST_F(LayoutConversionTest, PlanRefusesChangedRollRelation) {
  auto rolled = [&](ArrayRef<Attribute> dims, ArrayRef<int64_t> rolls) {
    return LayoutAttr::get(&context, ArrayAttr::get(&context, dims), 16,
                           DenseI64ArrayAttr::get(&context, rolls));
  };
  LayoutAttr zeroByOne = rolled({dim(0, 4), dim(1, 4)}, {0, 1});
  LayoutAttr oneByZero = rolled({dim(0, 4), dim(1, 4)}, {1, 0});
  EXPECT_TRUE(failed(rotom::planLayoutConversion(zeroByOne, oneByZero)));
  EXPECT_TRUE(failed(rotom::planLayoutConversion(oneByZero, zeroByOne)));
  // Dim 0 rolled by dim 1 again, but written against a swapped piece order:
  // the arguments no longer name the same positions, so this is refused.
  LayoutAttr swapped = rolled({dim(1, 4), dim(0, 4)}, {1, 0});
  EXPECT_TRUE(failed(rotom::planLayoutConversion(zeroByOne, swapped)));
}

TEST_F(LayoutConversionTest, EstimateIsNonzeroWhenGapPlacementDiffers) {
  LayoutAttr gapLast = layout({dim(0, 4), dim(/*dim=*/-2, 2)}, 8);
  LayoutAttr gapFirst = layout({dim(/*dim=*/-2, 2), dim(0, 4)}, 8);
  EXPECT_GT(rotom::estimateConversionCost(gapLast, gapFirst).rotations, 0);
}

TEST_F(LayoutConversionTest, ExpansionPureCtReplicationIsFreeCopies) {
  // 4x4 row-major (1 ct at n=16) expanded to 4 replicated ciphertexts:
  // every step is a full-row, zero-shift copy -- no rotations, no masks.
  LayoutAttr source = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr expanded = layout({dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)}, 16);
  auto steps = rotom::planLayoutConversion(source, expanded);
  ASSERT_TRUE(succeeded(steps));
  ASSERT_EQ(steps->steps.size(), 4u);
  for (const rotom::LayoutConversionStep& step : steps->steps) {
    EXPECT_EQ(step.sourceCt, 0);
    EXPECT_EQ(step.shift, 0);
    EXPECT_EQ(step.targetSlots.size(), 16u);
  }
}

// The estimate's mask and accumulate counts are per STEP, so they must
// match what the plan actually emits: every step masks its target slots,
// and every step after the first into a given ciphertext is accumulated.
// This holds exactly whenever the estimate gets the rotation count right.
TEST_F(LayoutConversionTest, EstimateMatchesThePlansMasksAndAccumulates) {
  struct Case {
    const char* name;
    LayoutAttr from;
    LayoutAttr to;
  };
  const Case cases[] = {
      // A trailing slot gap moves piece 0 from offset 1 to offset 2.
      {"new-gap", layout({dim(0, 4)}, 8),
       layout({dim(0, 4), dim(/*dim=*/-2, 2)}, 8)},
      // A slot transpose within one ciphertext.
      {"swap-slots", layout({dim(0, 4), dim(1, 4)}, 16),
       layout({dim(1, 4), dim(0, 4)}, 16)},
      // A swap across the ciphertext/slot boundary, four target ciphertexts.
      {"ct-to-slot", layout({dim(0, 4), dim(1, 4)}, 4),
       layout({dim(1, 4), dim(0, 4)}, 4)},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    auto plan = rotom::planLayoutConversion(c.from, c.to);
    ASSERT_TRUE(succeeded(plan));
    int64_t masks = 0;
    std::set<int64_t> targetCts;
    for (const rotom::LayoutConversionStep& step : plan->steps) {
      targetCts.insert(step.targetCt);
      if (static_cast<int64_t>(step.targetSlots.size()) != c.to.getN()) ++masks;
    }
    const int64_t accumulates = static_cast<int64_t>(plan->steps.size()) -
                                static_cast<int64_t>(targetCts.size());

    auto est = rotom::estimateConversionCost(c.from, c.to);
    EXPECT_EQ(est.masks, masks);
    EXPECT_EQ(est.accumulates, accumulates);
  }
}

// The price IS the plan: estimateConversionCost counts the steps
// planLayoutConversion produces, so the two cannot disagree. Expanding into
// slot replication is where a separate structural model used to go wrong --
// it priced one row's shifts while the plan writes each replica separately --
// so this pins the counts against the plan rather than merely bounding them.
TEST_F(LayoutConversionTest, EstimateCountsTheReplicatedScatterPlan) {
  LayoutAttr source = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr expanded = layout({dim(0, 4), dim(1, 4), dim(/*dim=*/-1, 4)}, 16);
  auto plan = rotom::planLayoutConversion(source, expanded);
  ASSERT_TRUE(succeeded(plan));
  int64_t masks = 0, accumulates = 0;
  llvm::DenseSet<std::pair<int64_t, int64_t>> rotations;
  llvm::DenseSet<int64_t> written;
  for (const rotom::LayoutConversionStep& step : plan->steps) {
    if (step.shift != 0) rotations.insert({step.sourceCt, step.shift});
    if (static_cast<int64_t>(step.targetSlots.size()) != expanded.getN())
      ++masks;
    if (!written.insert(step.targetCt).second) ++accumulates;
  }
  // A fill is log2(extent) rotate-and-add doublings, whatever the extent.
  int64_t fillDoublings = 0;
  for (const rotom::ReplicationFill& fill : plan->fills) {
    fillDoublings += llvm::Log2_64_Ceil(fill.extent);
  }
  auto est = rotom::estimateConversionCost(source, expanded);
  EXPECT_GT(masks, 0);  // the scatter this case is about
  EXPECT_EQ(est.masks, masks);
  EXPECT_EQ(est.rotations,
            static_cast<int64_t>(rotations.size()) + fillDoublings);
  EXPECT_EQ(est.accumulates, accumulates + fillDoublings);
}

TEST_F(LayoutConversionTest, ExpansionScatterNeedsRotationsAndMasks) {
  // Expanding 4x4 row-major so that ciphertext i holds row i replicated
  // ([0:4] to ct, [1:4] to slots, replication innermost): each target
  // ciphertext draws 4 slot-groups from the single source ciphertext, so
  // steps carry nonzero shifts and partial-row masks.
  LayoutAttr source = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr expanded = layout({dim(0, 4), dim(1, 4), dim(/*dim=*/-1, 4)}, 16);
  auto steps = rotom::planLayoutConversion(source, expanded);
  ASSERT_TRUE(succeeded(steps));
  EXPECT_GT(steps->steps.size(), 4u);
  bool sawShift = false;
  bool sawMask = false;
  for (const rotom::LayoutConversionStep& step : steps->steps) {
    if (step.shift != 0) sawShift = true;
    if (step.targetSlots.size() != 16u) sawMask = true;
  }
  EXPECT_TRUE(sawShift);
  EXPECT_TRUE(sawMask);
}

// The replicate-then-roll expansion: a column-major operand in one ciphertext
// expands onto roll(0,1) [k:ct];[R][i], where ciphertext c is the whole matrix
// with k shifted by c. The roll-by replication sits outermost in the slots and
// matches the source's outermost piece, so each target is one whole-ciphertext
// rotation -- no masks, no accumulates.
TEST_F(LayoutConversionTest, ReplicateThenRollExpansionIsPureRotations) {
  // Source: lhs (i, k) packed column-major, [1:4][0:4] (slot = 4k + i).
  LayoutAttr colMajor = layout({dim(1, 4), dim(0, 4)}, 16);
  // Target: [1:4];[-1:4][0:4] with rolls [(0, 1)] -- k on ct rolled by the
  // slot-outermost replication.
  LayoutAttr rolled = LayoutAttr::get(
      &context,
      ArrayAttr::get(&context, {dim(1, 4), dim(/*dim=*/-1, 4), dim(0, 4)}),
      /*n=*/16, DenseI64ArrayAttr::get(&context, {0, 1}));
  ASSERT_EQ(rotom::layoutNumCiphertexts(colMajor), 1);
  ASSERT_EQ(rotom::layoutNumCiphertexts(rolled), 4);

  auto steps = rotom::planLayoutConversion(colMajor, rolled);
  ASSERT_TRUE(succeeded(steps));
  // One full-row step per target ciphertext: shift 4c, no masks (all 16
  // slots), no accumulates (one step per target).
  ASSERT_EQ(steps->steps.size(), 4u);
  for (const rotom::LayoutConversionStep& step : steps->steps) {
    EXPECT_EQ(step.sourceCt, 0);
    EXPECT_EQ(step.shift, 4 * step.targetCt);
    EXPECT_EQ(step.targetSlots.size(), 16u);
  }
}

// Filling a rolled-by-replication placement from a replicated source is one
// whole-ciphertext rotation per NONZERO block shift: each block of the target
// holds the source rotated by its block index, and the source's replication
// lets every block draw from the replica that makes the shift uniform. The
// step plan expresses this directly, one group per distinct shift.
TEST_F(LayoutConversionTest, RolledFillPlansOneRotationPerNonzeroBlockShift) {
  constexpr int64_t kD = 8;
  constexpr int64_t kN = kD * kD;
  LayoutAttr source = layout({dim(/*dim=*/-1, kD), dim(0, kD)}, kN);
  LayoutAttr target = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, kD), dim(-1, kD)}), kN,
      DenseI64ArrayAttr::get(&context, {0, 1}));
  auto steps = rotom::planLayoutConversion(source, target);
  ASSERT_TRUE(succeeded(steps));
  std::set<int64_t> shifts;
  for (const rotom::LayoutConversionStep& step : steps->steps) {
    shifts.insert(step.shift);
  }
  shifts.erase(0);
  EXPECT_EQ(shifts.size(), static_cast<size_t>(kD - 1));
}

TEST_F(LayoutConversionTest, CompactionOfGappedMatmulResultPlansMaskedGathers) {
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

  auto steps = rotom::planLayoutConversion(source, compact);
  ASSERT_TRUE(succeeded(steps));
  ASSERT_EQ(steps->steps.size(), 4u);
  for (const rotom::LayoutConversionStep& step : steps->steps) {
    EXPECT_EQ(step.targetCt, 0);
    const int64_t i = step.sourceCt;
    EXPECT_EQ(step.shift, (16 - i) % 16);
    ASSERT_EQ(step.targetSlots.size(), 4u);
    for (int64_t j = 0; j < 4; ++j) {
      EXPECT_EQ(step.targetSlots[j], 4 * j + i);
    }
  }
}

// The step counts the retired relation-based planner produced, frozen as
// expectations: the structural planner reproduced every one of them exactly
// (same triples, same slot sets) before that planner was deleted.
TEST_F(LayoutConversionTest, PlansMatchTheRetiredRelationPlannerCounts) {
  struct Case {
    const char* name;
    LayoutAttr from;
    LayoutAttr to;
    size_t steps;
  };
  SmallVector<Case> cases = {
      {"ct-replication", layout({dim(0, 4), dim(1, 4)}, 16),
       layout({dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)}, 16), 4},
      {"rowmajor-to-colmajor", layout({dim(0, 4), dim(1, 4)}, 16),
       layout({dim(1, 4), dim(0, 4)}, 16), 7},
      {"gapped-to-compact",
       layout({dim(0, 4), dim(1, 4), dim(/*dim=*/-2, 4)}, 16),
       layout({dim(1, 4), dim(0, 4)}, 16), 4},
      // Large enough that point enumeration and group enumeration diverge:
      // 63 groups against 1024 tensor points.
      {"large-transpose", layout({dim(0, 32), dim(1, 32)}, 1024),
       layout({dim(1, 32), dim(0, 32)}, 1024), 63},
      {"reslot-split", layout({dim(0, 4), dim(1, 16)}, 64),
       layout({dim(1, 16), dim(0, 4)}, 64), 31},
      {"ct-reshuffle", layout({dim(0, 2), dim(1, 2), dim(2, 4)}, 4),
       layout({dim(1, 2), dim(0, 2), dim(2, 4)}, 4), 4},
  };
  for (const Case& c : cases) {
    auto plan = rotom::planLayoutConversion(c.from, c.to);
    ASSERT_TRUE(succeeded(plan)) << c.name;
    EXPECT_EQ(plan->steps.size(), c.steps) << c.name;
  }
}

// One added roll is a plan; two at once, or a removed roll, is not.
TEST_F(LayoutConversionTest, RefusesMoreThanOneRollChange) {
  LayoutAttr plain = layout({dim(0, 4), dim(1, 4), dim(2, 4)}, 64);
  LayoutAttr twoRolls = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, 4), dim(1, 4), dim(2, 4)}), 64,
      DenseI64ArrayAttr::get(&context, {1, 0, 2, 0}));
  EXPECT_TRUE(failed(rotom::planLayoutConversion(plain, twoRolls)));
  EXPECT_TRUE(failed(rotom::planLayoutConversion(twoRolls, plain)));
}

TEST_F(LayoutConversionTest, RefusesMismatchedCapacity) {
  LayoutAttr from = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr to = layout({dim(0, 4), dim(1, 4)}, 4);
  EXPECT_TRUE(failed(rotom::planLayoutConversion(from, to)));
}

// The MNIST layer-1 roll at n = 32768: 16 targets, each the one source
// ciphertext rotated by 64 * c, no fills. That is a roll by the ciphertext
// piece, which a matmul folds into baby-step/giant-step.
TEST_F(LayoutConversionTest, BsgsScheduleRecognizesTheMnistRoll) {
  LayoutAttr replicated = layout(
      {dim(-1, 16, 1), dim(-1, 32, 1024), dim(0, 1024), dim(1, 1)}, 32768);
  SmallVector<Attribute> rolledDims = {dim(0, 16, 64), dim(-1, 32, 16),
                                       dim(-1, 16, 1), dim(0, 64), dim(1, 1)};
  LayoutAttr rolled = LayoutAttr::get(
      &context, ArrayAttr::get(&context, rolledDims), 32768,
      DenseI64ArrayAttr::get(&context, {rotom::encodeRollArg({false, 0}),
                                        rotom::encodeRollArg({false, 2})}));
  auto roll = rotom::bsgsScheduleOpt(replicated, rolled);
  ASSERT_TRUE(roll.has_value());
  EXPECT_EQ(roll->stride, 64);
  EXPECT_EQ(roll->targets, 16);
  EXPECT_FALSE(roll->negative);
  // A conversion that fills is not a pure roll.
  LayoutAttr image =
      layout({dim(-1, 32, 1024), dim(0, 1024), dim(1, 1)}, 32768);
  LayoutAttr gappy = layout({dim(0, 512), dim(-2, 64), dim(1, 1)}, 32768);
  LayoutAttr filled = layout({dim(0, 512), dim(-1, 64), dim(1, 1)}, 32768);
  EXPECT_FALSE(rotom::bsgsScheduleOpt(gappy, filled).has_value());
}

}  // namespace
}  // namespace heir
}  // namespace mlir
