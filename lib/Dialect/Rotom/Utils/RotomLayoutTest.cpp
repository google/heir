#include <cstdint>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Utils/RotomLayout.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::RotomDialect;

class RotomLayoutTest : public ::testing::Test {
 protected:
  RotomLayoutTest() { context.loadDialect<RotomDialect>(); }

  DimAttr dim(int64_t dim, int64_t size, int64_t stride = 1) {
    return DimAttr::get(&context, dim, size, stride);
  }

  LayoutAttr layout(ArrayRef<Attribute> dims, int64_t n) {
    return LayoutAttr::get(&context, ArrayAttr::get(&context, dims), n);
  }

  MLIRContext context;
};

TEST_F(RotomLayoutTest, CountsSplitAxisByCiphertextPart) {
  // A 4x8 layout at n=16 whose row axis (extent 4) is split explicitly into a
  // high (ciphertext) piece [stride 2] and a low (slot) piece [stride 1] --
  // the explicit form of an axis spanning the ct/slot boundary. The column
  // axis (extent 8) and the row's low piece (extent 2) fill the 16 slots (8 * 2
  // == 16); the row's high piece (extent 2) indexes ciphertexts. The layout
  // therefore occupies 2 ciphertexts, not 4 (the full row extent).
  LayoutAttr split = layout({dim(0, 2, /*stride=*/2), dim(1, 8, /*stride=*/1),
                             dim(0, 2, /*stride=*/1)},
                            /*n=*/16);
  EXPECT_EQ(rotom::layoutNumCiphertexts(split), 2);
}

TEST_F(RotomLayoutTest, MaterializesSplitAxisLayout) {
  // A repeated dim id is a split of one tensor axis: the two pieces
  // of axis 0 (strides 2 and 1) and of axis 1 share their axis's domain
  // variable. This is a valid, materializable 2x2-tiled layout. (An invalid
  // split -- e.g. two stride-1 pieces -- is rejected by LayoutAttr::verify.)
  LayoutAttr split = layout({dim(0, 2, /*stride=*/2), dim(1, 2, /*stride=*/2),
                             dim(0, 2, /*stride=*/1), dim(1, 2, /*stride=*/1)},
                            /*n=*/8);
  EXPECT_TRUE(rotom::isMaterializableRotomLayout(split));
}

TEST_F(RotomLayoutTest, MergeAdjacentDimsJoinsContiguousPieces) {
  // [0:2:8][0:8:1] and [0:16:1] describe the same packing of a 16-vector: the
  // outer piece reads the index parts directly above the inner one.
  LayoutAttr split = layout({dim(0, 2, /*stride=*/8), dim(0, 8)}, 16);
  LayoutAttr whole = layout({dim(0, 16)}, 16);
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(split), whole);
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(whole), whole);
}

TEST_F(RotomLayoutTest, MergeAdjacentDimsJoinsReplicationAndGaps) {
  LayoutAttr replicated = layout({dim(-1, 2), dim(-1, 2), dim(0, 4)}, 16);
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(replicated),
            layout({dim(-1, 4), dim(0, 4)}, 16));
  LayoutAttr gapped = layout({dim(0, 4), dim(-2, 2), dim(-2, 2)}, 16);
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(gapped),
            layout({dim(0, 4), dim(-2, 4)}, 16));
}

TEST_F(RotomLayoutTest, MergeAdjacentDimsStopsAtCtSlotBoundary) {
  // The same two contiguous pieces, but the outer piece indexes ciphertexts
  // (2 * 8 > n = 8): the merged piece would straddle the ct/slot boundary, so
  // the form is already canonical.
  LayoutAttr straddling = layout({dim(0, 2, /*stride=*/8), dim(0, 8)}, 8);
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(straddling), straddling);
}

TEST_F(RotomLayoutTest, MergeAdjacentDimsSkipsNonContiguousPieces) {
  // [0:2:1][0:8:2] traverses the low part then the high parts -- not the
  // contiguous outer-above-inner order [0:2:8][0:8:1] merges.
  LayoutAttr interleaved =
      layout({dim(0, 2, /*stride=*/1), dim(0, 8, /*stride=*/2)}, 16);
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(interleaved), interleaved);
}

TEST_F(RotomLayoutTest, MergeAdjacentDimsPinsPieceRollArgs) {
  // roll(1, 0) reads piece 1's part of the index; joining pieces 1 and 2
  // (contiguous pieces of axis 0) would change what the roll rewrites.
  LayoutAttr rolled = LayoutAttr::getCanonical(
      &context, {dim(1, 4), dim(0, 2, /*stride=*/8), dim(0, 8)}, 64,
      /*rolls=*/{1, 0});
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(rolled), rolled);
}

TEST_F(RotomLayoutTest, MergeAdjacentDimsReindexesRollArgs) {
  // Pieces 0 and 1 merge, shifting the roll's piece arguments (2, 3) -> (1, 2).
  LayoutAttr split = LayoutAttr::getCanonical(
      &context, {dim(0, 2, /*stride=*/2), dim(0, 2), dim(1, 4), dim(2, 4)}, 64,
      /*rolls=*/{2, 3});
  LayoutAttr merged =
      LayoutAttr::getCanonical(&context, {dim(0, 4), dim(1, 4), dim(2, 4)}, 64,
                               /*rolls=*/{1, 2});
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(split), merged);
}

TEST_F(RotomLayoutTest, MergeAdjacentDimsRestatesAxisArgWhenUnsplit) {
  // An axis FROM rewrites the whole axis index, so its pieces may merge; once
  // axis 0 is a single piece the argument restates as that piece (the
  // canonical form of an unsplit axis).
  LayoutAttr split = LayoutAttr::getCanonical(
      &context, {dim(1, 16), dim(0, 2, /*stride=*/8), dim(0, 8)}, 256,
      /*rolls=*/{rotom::encodeRollArg({/*isAxis=*/true, 0}), 0});
  LayoutAttr merged = LayoutAttr::getCanonical(
      &context, {dim(1, 16), dim(0, 16)}, 256, /*rolls=*/{1, 0});
  EXPECT_EQ(rotom::mergeAdjacentLayoutDims(split), merged);
}

TEST_F(RotomLayoutTest, LowerableAllowsReplicationStride) {
  // A replication piece's stride describes replica placement.
  LayoutAttr replicated =
      layout({dim(0, 4), dim(/*dim=*/-1, /*size=*/2, /*stride=*/4)}, 8);
  EXPECT_TRUE(rotom::isLowerableRotomLayout(replicated));
}

TEST_F(RotomLayoutTest, LowerableAllowsSplitAxis) {
  // A split axis: the high piece's stride is its offset, which
  // the verifier requires to be the product of the lower extents. The
  // matmul pipeline produces such results (a ciphertext piece over a slot
  // piece), and they lower like any other layout.
  LayoutAttr split = layout({dim(0, 2, /*stride=*/2), dim(0, 2), dim(1, 4)}, 8);
  EXPECT_TRUE(rotom::isLowerableRotomLayout(split));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
