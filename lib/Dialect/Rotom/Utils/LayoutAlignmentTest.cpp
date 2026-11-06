#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/LayoutAlignment.h"
#include "lib/Dialect/Rotom/Utils/LayoutConversion.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
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

// Operator alignment, matmul map: lhs is A(i, k) with i innermost, rhs is
// B(k, j) traversing k in lockstep and replicating where lhs traverses i.
TEST_F(LayoutAlignmentTest, OperatorAlignedMatmulOperands) {
  rotom::OperatorAlignmentMap map = rotom::OperatorAlignmentMap::matmul();
  // lhs dims: i = 0, k = 1. rhs dims: k = 0, j = 1.
  LayoutAttr lhsPacked = layout({dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr rhsPacked = layout({dim(0, 4), dim(/*dim=*/-1, 4)}, 16);
  EXPECT_TRUE(rotom::isOperatorAligned(map, lhsPacked, rhsPacked));
  // Swapping rhs's pieces breaks the positional correspondence.
  LayoutAttr rhsSwapped = layout({dim(/*dim=*/-1, 4), dim(0, 4)}, 16);
  EXPECT_FALSE(rotom::isOperatorAligned(map, lhsPacked, rhsSwapped));
}

// The matmul map by rank reproduces the reference's per-shape alignment
// table: the 2-D default is {0:R, 1:0, R:1}; mat-vec and vec-mat drop the
// missing free dim; block matmul locksteps the shared batch dim; a batched
// lhs against a 2-D rhs pairs its batch dim with replication.
TEST_F(LayoutAlignmentTest, MatmulMapGeneralizesOverOperandRanks) {
  using Map = rotom::OperatorAlignmentMap;
  const int64_t R = Map::kRepeatedDim;
  auto check = [](const Map& map, ArrayRef<int64_t> lhsToRhs,
                  ArrayRef<int64_t> rhsToLhs) {
    EXPECT_EQ(SmallVector<int64_t>(map.lhsToRhs),
              SmallVector<int64_t>(lhsToRhs));
    EXPECT_EQ(SmallVector<int64_t>(map.rhsToLhs),
              SmallVector<int64_t>(rhsToLhs));
  };
  check(Map::matmul(), {R, 0}, {1, R});
  check(Map::matmul(2, 2), {R, 0}, {1, R});
  check(Map::matmul(2, 1), {R, 0}, {1});
  check(Map::matmul(1, 2), {0}, {0, R});
  check(Map::matmul(1, 1), {0}, {0});
  check(Map::matmul(3, 3), {0, R, 1}, {0, 2, R});
  check(Map::matmul(3, 2), {R, R, 0}, {2, R});
  check(Map::matmul(2, 3), {R, 1}, {R, 1, R});
  check(Map::matmul(4, 3), {R, 0, R, 1}, {1, 3, R});
}

// Mat-vec with the map for ranks (2, 1): A(i, k) traverses k where x(k)
// traverses k and replicates where A traverses i.
TEST_F(LayoutAlignmentTest, OperatorAlignedMatVecOperands) {
  auto map = rotom::OperatorAlignmentMap::matmul(2, 1);
  LayoutAttr lhs = layout({dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr rhs = layout({dim(0, 4), dim(/*dim=*/-1, 4)}, 16);
  EXPECT_TRUE(rotom::isOperatorAligned(map, lhs, rhs));
  LayoutAttr rhsSwapped = layout({dim(/*dim=*/-1, 4), dim(0, 4)}, 16);
  EXPECT_FALSE(rotom::isOperatorAligned(map, lhs, rhsSwapped));
}

// Block matmul with the map for ranks (3, 3): the batch dim locksteps on
// both sides, then the 2-D pattern applies to the inner dims.
TEST_F(LayoutAlignmentTest, OperatorAlignedBlockMatmulOperands) {
  auto map = rotom::OperatorAlignmentMap::matmul(3, 3);
  // lhs dims: b = 0, i = 1, k = 2. rhs dims: b = 0, k = 1, j = 2.
  LayoutAttr lhs = layout({dim(0, 2), dim(2, 4), dim(1, 4)}, 32);
  LayoutAttr rhs = layout({dim(0, 2), dim(1, 4), dim(/*dim=*/-1, 4)}, 32);
  EXPECT_TRUE(rotom::isOperatorAligned(map, lhs, rhs));
  // A batch dim paired with the wrong dim is misaligned.
  LayoutAttr rhsBad = layout({dim(/*dim=*/-1, 2), dim(1, 4), dim(0, 4)}, 32);
  EXPECT_FALSE(rotom::isOperatorAligned(map, lhs, rhsBad));
}

// A k-diagonal needs the matching roll on BOTH operands: the rolled dim (k)
// pairs with a traversal dim on the other side.
TEST_F(LayoutAlignmentTest, OperatorAlignmentRequiresMatchingContractionRoll) {
  rotom::OperatorAlignmentMap map = rotom::OperatorAlignmentMap::matmul();
  LayoutAttr lhsDiag = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(1, 4), dim(0, 4)}), /*n=*/16,
      DenseI64ArrayAttr::get(&context, {0, 1}));
  LayoutAttr rhsDiag = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, 4), dim(/*dim=*/-1, 4)}),
      /*n=*/16, DenseI64ArrayAttr::get(&context, {0, 1}));
  LayoutAttr rhsPlain = layout({dim(0, 4), dim(/*dim=*/-1, 4)}, 16);
  EXPECT_TRUE(rotom::isOperatorAligned(map, lhsDiag, rhsDiag));
  EXPECT_FALSE(rotom::isOperatorAligned(map, lhsDiag, rhsPlain));
}

// A roll whose FROM piece pairs with replication on the other side is exempt:
// every index of the replicated side reads the same value, so rolling the
// traversal side cannot misalign the pairing.
TEST_F(LayoutAlignmentTest, OperatorAlignmentExemptsRollOntoReplication) {
  rotom::OperatorAlignmentMap map = rotom::OperatorAlignmentMap::matmul();
  // FROM = the i piece (position 1), BY = the k piece (position 0).
  LayoutAttr lhsRolled = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(1, 4), dim(0, 4)}), /*n=*/16,
      DenseI64ArrayAttr::get(&context, {1, 0}));
  LayoutAttr rhsPlain = layout({dim(0, 4), dim(/*dim=*/-1, 4)}, 16);
  EXPECT_TRUE(rotom::isOperatorAligned(map, lhsRolled, rhsPlain));
}

// alignRolls is the constructive form of isOperatorAligned: where the checker
// says "not aligned", this says which rolls each side must add.
TEST_F(LayoutAlignmentTest, AlignRollsReportsNothingWhenAlreadyAligned) {
  LayoutAttr a = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, 4), dim(1, 4)}), 16,
      DenseI64ArrayAttr::get(&context, {1, 0}));
  auto map = rotom::OperatorAlignmentMap::identity(2);
  ASSERT_TRUE(rotom::isOperatorAligned(map, a, a));
  auto alignment = rotom::alignRolls(map, a, a);
  ASSERT_TRUE(succeeded(alignment));
  EXPECT_TRUE(alignment->empty());
}

TEST_F(LayoutAlignmentTest, AlignRollsAddsTheMissingRollToTheUnrolledSide) {
  LayoutAttr rolled = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, 4), dim(1, 4)}), 16,
      DenseI64ArrayAttr::get(&context, {1, 0}));
  LayoutAttr plain = layout({dim(0, 4), dim(1, 4)}, 16);
  auto map = rotom::OperatorAlignmentMap::identity(2);
  EXPECT_FALSE(rotom::isOperatorAligned(map, rolled, plain));

  auto alignment = rotom::alignRolls(map, rolled, plain);
  ASSERT_TRUE(succeeded(alignment));
  // The rolled side needs nothing; the plain side must take the roll.
  EXPECT_TRUE(alignment->addToLhs.empty());
  ASSERT_EQ(alignment->addToRhs.size(), 1u);
  EXPECT_FALSE(alignment->addToRhs[0].from.isAxis);
  EXPECT_EQ(alignment->addToRhs[0].from.index, 1);
  EXPECT_FALSE(alignment->addToRhs[0].by.isAxis);
  EXPECT_EQ(alignment->addToRhs[0].by.index, 0);

  // The report is symmetric under swapping the operands.
  auto swapped = rotom::alignRolls(map, plain, rolled);
  ASSERT_TRUE(succeeded(swapped));
  EXPECT_TRUE(swapped->addToRhs.empty());
  ASSERT_EQ(swapped->addToLhs.size(), 1u);
}

TEST_F(LayoutAlignmentTest, AlignRollsFailsWhenPieceStructuresDisagree) {
  // No roll change repairs a placement disagreement.
  LayoutAttr rowMajor = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr colMajor = layout({dim(1, 4), dim(0, 4)}, 16);
  auto map = rotom::OperatorAlignmentMap::identity(2);
  EXPECT_TRUE(failed(rotom::alignRolls(map, rowMajor, colMajor)));
}

// ---- alignment engine -----------------------------------------------------

// Piece list as text, for readable failures: [d:size:stride] with R and G.
static std::string fmtDims(ArrayRef<rotom::DimAttr> dims) {
  std::string out;
  for (rotom::DimAttr d : dims) {
    out += "[";
    out += d.isReplicate() ? "R" : d.isGap() ? "G" : std::to_string(d.getDim());
    out += ":" + std::to_string(d.getSize()) + ":" +
           std::to_string(d.getStride()) + "]";
  }
  return out;
}
static std::string fmtRolls(LayoutAttr layout) {
  std::string out;
  for (const rotom::RollSpec& r : rotom::getRollSpecs(layout)) {
    auto arg = [](const rotom::RollArg& a) {
      return (a.isAxis ? "axis " : "") + std::to_string(a.index);
    };
    out += "(" + arg(r.from) + "," + arg(r.by) + ")";
  }
  return out;
}

// The paper's A[i,k] x B[k,j] at 4x4, n = 16: A column-major and B row-major,
// one ciphertext each. The aligned shape is i*k*j = 64, so each side gains a
// factor of 4 as a new outermost (ciphertext) replication.
TEST_F(LayoutAlignmentTest, ReplicateForAlignmentAddsOuterReplication) {
  LayoutAttr a = layout({dim(1, 4), dim(0, 4)}, 16);  // A: slot = 4k + i
  LayoutAttr b = layout({dim(0, 4), dim(1, 4)}, 16);  // B: slot = 4k + j
  auto map = rotom::OperatorAlignmentMap::matmul();
  auto pair = rotom::replicateForAlignment(map, a, b, {4, 4}, {4, 4});
  ASSERT_TRUE(succeeded(pair));
  EXPECT_EQ(fmtDims(rotom::layoutDims(pair->lhs)), "[R:4:1][1:4:1][0:4:1]");
  EXPECT_EQ(fmtDims(rotom::layoutDims(pair->rhs)), "[R:4:1][0:4:1][1:4:1]");
  EXPECT_EQ(rotom::layoutNumCiphertexts(pair->lhs), 4);
  EXPECT_EQ(rotom::layoutNumCiphertexts(pair->rhs), 4);
}

// apply_sum_roll: the replicated lhs has replication on the ciphertext side
// and k in the slots; the two swap and k (now a ciphertext piece) rolls by
// the replication (now a slot piece).
TEST_F(LayoutAlignmentTest, ApplySumRollSwapsAndRollsTheSummationDim) {
  LayoutAttr replicated =
      layout({dim(/*dim=*/-1, 4), dim(1, 4), dim(0, 4)}, 16);
  std::optional<LayoutAttr> rolled =
      rotom::applySumRoll(replicated, /*sumDim=*/1);
  ASSERT_TRUE(rolled.has_value());
  EXPECT_EQ(fmtDims(rotom::layoutDims(*rolled)), "[1:4:1][R:4:1][0:4:1]");
  EXPECT_EQ(fmtRolls(*rolled), "(0,1)");
  EXPECT_EQ(rotom::layoutNumCiphertexts(*rolled), 4);

  // The symmetric rhs: k is dim 0 there.
  LayoutAttr rhs = layout({dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)}, 16);
  std::optional<LayoutAttr> rolledRhs = rotom::applySumRoll(rhs, /*sumDim=*/0);
  ASSERT_TRUE(rolledRhs.has_value());
  EXPECT_EQ(fmtDims(rotom::layoutDims(*rolledRhs)), "[0:4:1][R:4:1][1:4:1]");
  EXPECT_EQ(fmtRolls(*rolledRhs), "(0,1)");
}

TEST_F(LayoutAlignmentTest, ApplySumRollRefusesWithoutCiphertextReplication) {
  // No replication on the ciphertext side: nothing to roll by.
  LayoutAttr a = layout({dim(1, 4), dim(0, 4)}, 16);
  EXPECT_FALSE(rotom::applySumRoll(a, /*sumDim=*/1).has_value());
  // sumDim not in the slots.
  LayoutAttr b = layout({dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)}, 16);
  EXPECT_FALSE(rotom::applySumRoll(b, /*sumDim=*/5).has_value());
}

// The whole reference pipeline for one operand: replicate, then the sum roll.
TEST_F(LayoutAlignmentTest, SumRollFromReplicatedSourceIsPureRotations) {
  LayoutAttr replicated =
      layout({dim(/*dim=*/-1, 4), dim(1, 4), dim(0, 4)}, 16);
  std::optional<LayoutAttr> rolled =
      rotom::applySumRoll(replicated, /*sumDim=*/1);
  ASSERT_TRUE(rolled.has_value());
  auto plan = rotom::planLayoutConversion(replicated, *rolled);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->steps.size(), 4u);
  EXPECT_TRUE(plan->fills.empty());
  for (const rotom::LayoutConversionStep& step : plan->steps) {
    EXPECT_EQ(step.shift, 4 * step.targetCt);
    EXPECT_EQ(step.targetSlots.size(), 16u);
  }
}

// alignedDims mirrors each side's structure onto the other's tensor. For the
// replicated matmul pair, lhs [R][k][i] seen from rhs becomes [j][k][R]: the
// lhs replication broadcasts over j, k pairs with k, and i is replication on
// the rhs side. Symmetrically rhs [R][k][j] seen from lhs is [i][k][R].
TEST_F(LayoutAlignmentTest, AlignedDimsMirrorsStructureThroughTheMap) {
  SmallVector<rotom::DimAttr> lhs = {dim(/*dim=*/-1, 4), dim(1, 4), dim(0, 4)};
  SmallVector<rotom::DimAttr> rhs = {dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)};
  auto map = rotom::OperatorAlignmentMap::matmul();
  auto aligned = rotom::alignedDims(map, lhs, rhs, &context);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_EQ(fmtDims(aligned->forRhs), "[1:4:1][0:4:1][R:4:1]");
  EXPECT_EQ(fmtDims(aligned->forLhs), "[0:4:1][1:4:1][R:4:1]");
}

// For an identity map the mirror is the relabeled copy, so a pair that is
// already aligned mirrors to itself.
TEST_F(LayoutAlignmentTest, AlignedDimsIsIdentityForAlignedElementwise) {
  SmallVector<rotom::DimAttr> a = {dim(0, 4), dim(1, 4)};
  auto map = rotom::OperatorAlignmentMap::identity(2);
  auto aligned = rotom::alignedDims(map, a, a, &context);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_EQ(fmtDims(aligned->forRhs), fmtDims(a));
  EXPECT_EQ(fmtDims(aligned->forLhs), fmtDims(a));
}

// rollToAlign: the replicated matmul pair is not aligned (i sits where the
// rhs has j). Rolling the rhs toward its mirror [j][k][R] moves j to the
// ciphertext side by rolling it by the replication that was there.
TEST_F(LayoutAlignmentTest, RollToAlignRollsTheOutOfPlacePieceBySwap) {
  LayoutAttr lhs = layout({dim(/*dim=*/-1, 4), dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr rhs = layout({dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)}, 16);
  auto map = rotom::OperatorAlignmentMap::matmul();
  ASSERT_FALSE(rotom::isOperatorAligned(map, lhs, rhs));

  auto cands = rotom::rollToAlign(map, lhs, rhs);
  ASSERT_EQ(cands.size(), 2u);
  // Rhs rolled toward the lhs.
  EXPECT_EQ(cands[0].lhs, lhs);
  EXPECT_EQ(fmtDims(rotom::layoutDims(cands[0].rhs)), "[1:4:1][0:4:1][R:4:1]");
  EXPECT_EQ(fmtRolls(cands[0].rhs), "(0,2)");
  EXPECT_TRUE(rotom::isOperatorAligned(map, cands[0].lhs, cands[0].rhs));
  // Lhs rolled toward the rhs.
  EXPECT_EQ(cands[1].rhs, rhs);
  EXPECT_EQ(fmtDims(rotom::layoutDims(cands[1].lhs)), "[0:4:1][1:4:1][R:4:1]");
  EXPECT_EQ(fmtRolls(cands[1].lhs), "(0,2)");
  EXPECT_TRUE(rotom::isOperatorAligned(map, cands[1].lhs, cands[1].rhs));
}

// alignPair on an unrolled pair offers both conversions and both rolls.
TEST_F(LayoutAlignmentTest, AlignPairOffersConversionsAndRolls) {
  LayoutAttr lhs = layout({dim(/*dim=*/-1, 4), dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr rhs = layout({dim(/*dim=*/-1, 4), dim(0, 4), dim(1, 4)}, 16);
  auto map = rotom::OperatorAlignmentMap::matmul();
  auto cands = rotom::alignPair(map, lhs, rhs);
  ASSERT_EQ(cands.size(), 4u);
  for (const rotom::AlignedPair& c : cands) {
    EXPECT_TRUE(rotom::isOperatorAligned(map, c.lhs, c.rhs));
  }
  // Conversions first: the converted side has no rolls.
  EXPECT_EQ(fmtDims(rotom::layoutDims(cands[0].rhs)), "[1:4:1][0:4:1][R:4:1]");
  EXPECT_EQ(fmtRolls(cands[0].rhs), "");
  EXPECT_EQ(fmtDims(rotom::layoutDims(cands[1].lhs)), "[0:4:1][1:4:1][R:4:1]");
  EXPECT_EQ(fmtRolls(cands[1].lhs), "");
  // Then the rolls.
  EXPECT_EQ(fmtRolls(cands[2].rhs), "(0,2)");
  EXPECT_EQ(fmtRolls(cands[3].lhs), "(0,2)");
}

TEST_F(LayoutAlignmentTest, AlignPairReturnsAlignedPairUnchanged) {
  LayoutAttr a = layout({dim(0, 4), dim(1, 4)}, 16);
  auto map = rotom::OperatorAlignmentMap::identity(2);
  auto cands = rotom::alignPair(map, a, a);
  ASSERT_EQ(cands.size(), 1u);
  EXPECT_EQ(cands[0].lhs, a);
  EXPECT_EQ(cands[0].rhs, a);
}

// Elementwise row-major vs column-major: two conversions. The roll
// candidates are filtered out -- for an identity map a roll on one side
// pairs with a traversal dim on the other, so it is required and unmatched.
TEST_F(LayoutAlignmentTest, AlignPairElementwiseOffersOnlyConversions) {
  LayoutAttr rowMajor = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr colMajor = layout({dim(1, 4), dim(0, 4)}, 16);
  auto map = rotom::OperatorAlignmentMap::identity(2);
  auto cands = rotom::alignPair(map, rowMajor, colMajor);
  ASSERT_EQ(cands.size(), 2u);
  EXPECT_EQ(cands[0].lhs, rowMajor);
  EXPECT_EQ(cands[0].rhs, rowMajor);
  EXPECT_EQ(cands[1].lhs, colMajor);
  EXPECT_EQ(cands[1].rhs, colMajor);
}

// outputLayout, matmul: the summation piece becomes a gap, a replication
// paired with a traversal dim becomes that dim, and a surviving roll carries
// over at its position.
TEST_F(LayoutAlignmentTest, OutputLayoutMatmulConsumesTheSummationDim) {
  auto map = rotom::OperatorAlignmentMap::matmul();
  LayoutAttr lhs = layout({dim(/*dim=*/-1, 4), dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr convRhs = layout({dim(1, 4), dim(0, 4), dim(/*dim=*/-1, 4)}, 16);
  auto out = rotom::outputLayout(map, /*isMatmul=*/true, lhs, convRhs,
                                 /*lhsSumDim=*/1, /*rhsSumDim=*/0);
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(fmtDims(rotom::layoutDims(*out)), "[1:4:1][G:4:1][0:4:1]");
  EXPECT_EQ(fmtRolls(*out), "");

  // The rolled rhs candidate: its roll's FROM is j, not k, so it survives
  // positionally -- j (ciphertext) rolled by the piece now at position 2, i.
  LayoutAttr rolledRhs = LayoutAttr::get(
      &context,
      ArrayAttr::get(&context, {dim(1, 4), dim(0, 4), dim(/*dim=*/-1, 4)}), 16,
      DenseI64ArrayAttr::get(&context, {0, 2}));
  auto outRolled = rotom::outputLayout(map, true, lhs, rolledRhs, 1, 0);
  ASSERT_TRUE(outRolled.has_value());
  EXPECT_EQ(fmtDims(rotom::layoutDims(*outRolled)), "[1:4:1][G:4:1][0:4:1]");
  EXPECT_EQ(fmtRolls(*outRolled), "(0,2)");
}

TEST_F(LayoutAlignmentTest, OutputLayoutElementwiseIsTheLhs) {
  auto map = rotom::OperatorAlignmentMap::identity(2);
  LayoutAttr a = layout({dim(0, 4), dim(1, 4)}, 16);
  auto out = rotom::outputLayout(map, /*isMatmul=*/false, a, a, -1, -1);
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(*out, a);
}

// The whole reference matmul pipeline.
TEST_F(LayoutAlignmentTest, ReferenceMatmulPipelineYieldsDiagonalResult) {
  auto map = rotom::OperatorAlignmentMap::matmul();
  LayoutAttr a = layout({dim(1, 4), dim(0, 4)}, 16);
  LayoutAttr b = layout({dim(0, 4), dim(1, 4)}, 16);
  auto rep = rotom::replicateForAlignment(map, a, b, {4, 4}, {4, 4});
  ASSERT_TRUE(succeeded(rep));
  auto lhs = rotom::applySumRoll(rep->lhs, /*sumDim=*/1);
  auto rhs = rotom::applySumRoll(rep->rhs, /*sumDim=*/0);
  ASSERT_TRUE(lhs.has_value());
  ASSERT_TRUE(rhs.has_value());
  EXPECT_FALSE(rotom::isOperatorAligned(map, *lhs, *rhs));

  auto cands = rotom::alignPair(map, *lhs, *rhs);
  ASSERT_FALSE(cands.empty());
  for (const rotom::AlignedPair& c : cands) {
    EXPECT_TRUE(rotom::isOperatorAligned(map, c.lhs, c.rhs));
  }
  // The rhs rolled toward the lhs: j takes the replication's place and rolls
  // by it, on top of the sum roll. The repack candidates share the list, so
  // the rolled one is found by its shape.
  const rotom::AlignedPair* rolledPair = nullptr;
  for (const rotom::AlignedPair& c : cands) {
    if (fmtRolls(c.rhs) == "(0,1)(1,2)") rolledPair = &c;
  }
  ASSERT_NE(rolledPair, nullptr);
  EXPECT_EQ(fmtDims(rotom::layoutDims(rolledPair->rhs)),
            "[0:4:1][1:4:1][R:4:1]");

  auto out =
      rotom::outputLayout(map, true, rolledPair->lhs, rolledPair->rhs, 1, 0);
  ASSERT_TRUE(out.has_value());
  // The summation dim sat in the ciphertext region, so the reduction removed
  // it rather than leaving a gap behind, and the roll's pieces moved down.
  EXPECT_EQ(fmtDims(rotom::layoutDims(*out)), "[1:4:1][0:4:1]");
  EXPECT_EQ(fmtRolls(*out), "(0,1)");
  EXPECT_EQ(rotom::layoutNumCiphertexts(*out), 1);
}

// The Rotom reference's MNIST layer-1 plan at n = 32768.
TEST_F(LayoutAlignmentTest, MatchPublicReachesReferenceMnistPlan) {
  auto map = rotom::OperatorAlignmentMap::matmul();
  // Weights (i = 0, k = 1) over 16 ciphertexts; image (k = 0, j = 1), one.
  LayoutAttr lhs = layout({dim(0, 16, 32), dim(0, 32), dim(1, 1024)}, 32768);
  LayoutAttr rhs =
      layout({dim(/*dim=*/-1, 32, 1024), dim(0, 1024), dim(1, 1)}, 32768);
  SmallVector<int64_t> lhsShape = {512, 1024}, rhsShape = {1024, 1};
  auto rep = rotom::replicateForAlignment(map, lhs, rhs, lhsShape, rhsShape);
  ASSERT_TRUE(succeeded(rep));
  std::optional<LayoutAttr> rolled = rotom::applySumRoll(rep->rhs, 0);
  ASSERT_TRUE(rolled.has_value());
  // The weights have no ciphertext replication, so they cannot take the roll
  // themselves; only the repack reaches the aligned pair.
  EXPECT_FALSE(rotom::applySumRoll(rep->lhs, 1).has_value());
  auto pairs = rotom::alignPair(map, rep->lhs, *rolled);
  ASSERT_FALSE(pairs.empty());
  bool sawRolledPair = false;
  for (const auto& p : pairs) {
    if (p.rhs != *rolled) continue;
    if (!p.lhs.getRolls() || p.lhs.getRolls().empty()) continue;
    EXPECT_EQ(p.lhs.getRolls(), rolled->getRolls());
    EXPECT_TRUE(rotom::isOperatorAligned(map, p.lhs, p.rhs));
    EXPECT_TRUE(rotom::outputLayout(map, /*isMatmul=*/true, p.lhs, p.rhs,
                                    /*lhsSumDim=*/1, /*rhsSumDim=*/0)
                    .has_value());
    sawRolledPair = true;
  }
  EXPECT_TRUE(sawRolledPair);
}

// matchPublicLayout copies the other side's rolls by position, so a
// whole-axis roll argument has no counterpart and the repack is refused.
TEST_F(LayoutAlignmentTest, MatchPublicRefusesAxisNamedRolls) {
  auto map = rotom::OperatorAlignmentMap::matmul();
  LayoutAttr lhs = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr rhs = LayoutAttr::get(
      &context, ArrayAttr::get(&context, {dim(0, 2, 2), dim(0, 2), dim(1, 4)}),
      16,
      DenseI64ArrayAttr::get(&context,
                             {rotom::encodeRollArg({/*isAxis=*/true, 0}),
                              rotom::encodeRollArg({/*isAxis=*/false, 2})}));
  EXPECT_FALSE(
      rotom::matchPublicLayout(map, lhs, rhs, /*matchLhs=*/true).has_value());
}

// A replication piece wider than the dim it broadcasts over: the second MNIST
// layer at n = 32768 replicates the 10x512 weights 4x to reach the aligned
// fill, and the mirror maps that R:4 onto j, whose extent is 1. The leftover
// must stay replication or the mirrored layout no longer fills the
// ciphertext and every candidate dies.
TEST_F(LayoutAlignmentTest, AlignedDimsKeepsReplicationBeyondUnitDim) {
  SmallVector<rotom::DimAttr> lhs = {dim(/*dim=*/-1, 4), dim(0, 16),
                                     dim(1, 512)};
  SmallVector<rotom::DimAttr> rhs = {dim(/*dim=*/-1, 64, /*stride=*/512),
                                     dim(0, 512), dim(1, 1)};
  auto map = rotom::OperatorAlignmentMap::matmul();
  auto aligned = rotom::alignedDims(map, lhs, rhs, &context);
  ASSERT_TRUE(aligned.has_value());
  int64_t capacity = 1;
  for (rotom::DimAttr d : aligned->forRhs) capacity *= d.getSize();
  EXPECT_EQ(capacity, 4 * 16 * 512);
  LayoutAttr lhsLayout =
      layout({dim(/*dim=*/-1, 4), dim(0, 16), dim(1, 512)}, 32768);
  LayoutAttr rhsLayout = layout(
      {dim(/*dim=*/-1, 64, /*stride=*/512), dim(0, 512), dim(1, 1)}, 32768);
  EXPECT_FALSE(rotom::alignPair(map, lhsLayout, rhsLayout).empty());
}

}  // namespace
}  // namespace heir
}  // namespace mlir
