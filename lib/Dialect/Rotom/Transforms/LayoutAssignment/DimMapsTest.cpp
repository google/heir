#include <cstdint>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/DimMaps.h"
#include "llvm/include/llvm/ADT/SmallVector.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using rotom::DimAttr;
using rotom::LayoutAttr;
using rotom::RotomDialect;

class DimMapsTest : public ::testing::Test {
 protected:
  DimMapsTest() { context.loadDialect<RotomDialect>(); }

  DimAttr dim(int64_t dim, int64_t size, int64_t stride = 1) {
    return DimAttr::get(&context, dim, size, stride);
  }

  LayoutAttr layout(ArrayRef<Attribute> dims, int64_t n) {
    return LayoutAttr::getCanonical(&context, asDims(dims), n);
  }

  LayoutAttr rolled(ArrayRef<Attribute> dims, int64_t n,
                    ArrayRef<int64_t> rolls) {
    return LayoutAttr::getCanonical(&context, asDims(dims), n, rolls);
  }

  SmallVector<DimAttr> asDims(ArrayRef<Attribute> dims) {
    SmallVector<DimAttr> vec;
    for (Attribute attr : dims) vec.push_back(cast<DimAttr>(attr));
    return vec;
  }

  MLIRContext context;
};

// A rolled candidate remapped through an identity dim map keeps its roll --
// dropping it would misdescribe the diagonal packing as plain.
TEST_F(DimMapsTest, RemapCarriesRolls) {
  LayoutAttr in = rolled({dim(0, 4), dim(1, 4)}, 16, /*rolls=*/{1, 0});
  LayoutAttr out = rotom::remapLayoutDims(in, /*oldToNewDim=*/{0, 1});
  EXPECT_EQ(out, in);
}

// Relabeling dims (dim 0 -> 1, dim 1 -> 0, i.e. a transpose) remaps the roll's
// piece positions consistently; the pieces keep their list order so the roll
// positions are unchanged, but the dim ids swap.
TEST_F(DimMapsTest, RemapRelabelsRolledDims) {
  LayoutAttr in = rolled({dim(0, 4), dim(1, 4)}, 16, /*rolls=*/{1, 0});
  LayoutAttr out = rotom::remapLayoutDims(in, /*oldToNewDim=*/{1, 0});
  LayoutAttr expected = rolled({dim(1, 4), dim(0, 4)}, 16, /*rolls=*/{1, 0});
  EXPECT_EQ(out, expected);
}

// A roll whose piece the op drops cannot be represented on the result, so the
// remap fails (the caller then drops the candidate) rather than emitting a
// plain layout that hides the diagonal.
TEST_F(DimMapsTest, RemapFailsWhenRolledDimDropped) {
  // roll (from=pos1=dim1, by=pos0=dim0); dropping dim 0 removes the by-piece.
  LayoutAttr in = rolled({dim(0, 4), dim(1, 4)}, 16, /*rolls=*/{1, 0});
  LayoutAttr out = rotom::remapLayoutDims(in, /*oldToNewDim=*/{-1, 0});
  EXPECT_EQ(out, nullptr);
}

// An axis roll argument names a tensor axis, so a relabeling remaps it
// through the dim map (piece arguments travel positionally).
TEST_F(DimMapsTest, RemapRelabelsAxisRollArgs) {
  const int64_t axis0 = rotom::encodeRollArg({/*isAxis=*/true, 0});
  const int64_t axis1 = rotom::encodeRollArg({/*isAxis=*/true, 1});
  // dim 0 split into two pieces, rolled whole by the dim 1 piece.
  LayoutAttr in = rolled({dim(0, 4, /*stride=*/4), dim(0, 4), dim(1, 16)}, 16,
                         /*rolls=*/{axis0, 2});
  LayoutAttr out = rotom::remapLayoutDims(in, /*oldToNewDim=*/{1, 0});
  LayoutAttr expected =
      rolled({dim(1, 4, /*stride=*/4), dim(1, 4), dim(0, 16)}, 16,
             /*rolls=*/{axis1, 2});
  EXPECT_EQ(out, expected);

  // Dropping the rolled axis makes the roll unrepresentable: remap fails.
  EXPECT_EQ(rotom::remapLayoutDims(in, /*oldToNewDim=*/{-1, 0}), nullptr);
}

// A roll-free candidate passes through the remap unchanged.
TEST_F(DimMapsTest, RemapRollFreeUnchanged) {
  LayoutAttr in = layout({dim(0, 4), dim(1, 4)}, 16);
  LayoutAttr out = rotom::remapLayoutDims(in, /*oldToNewDim=*/{0, 1});
  EXPECT_EQ(out, in);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
