#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;

TEST(ConvolutionTest, ConvFilterRelation) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  int64_t padding = 1;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 8);

  // Handwritten expected relation
  auto relation = getIntegerRelationFromIslStr(
      "{ [ifr, ifc] -> [mr, mc] : exists idr, idc : -1 <= idr and idr <= 1 and "
      "-1 <= idc and idc <= 1 and 0 <= ifr and ifr <= 2 and 0 <= ifc and ifc "
      "<= 2 and mr = idc + 1 + 3 * (idr + 1) and mc = -4 + mr + ifc + "
      "ifr * 3 and 0 <= idr + ifr and idr + ifr <= 2 and 0 <= idc + ifc and "
      "idc + ifc <= 2 }");
  relation.value().simplify();
  ASSERT_TRUE(succeeded(relation));
  EXPECT_TRUE(convFilterRelation.isEqual(relation.value()));
}

TEST(ConvolutionTest, ConvFilterRelationNoPadding) {
  // No padding and same size should result in a single multiplication of the
  // two flattened inputs.
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  int64_t padding = 0;
  SmallVector<int64_t> strides = {1, 1};
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 0);
}

TEST(ConvolutionTest, ConvFilterRelation4x4Data) {
  // No padding on a larger data matrix should result in 4 ciphertexts.
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({4, 4}, IndexType::get(&context));
  int64_t padding = 0;
  SmallVector<int64_t> strides = {1, 1};
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 3);
}

TEST(ConvolutionTest, ConvFilterRelationPadding2) {
  // Two padding on a larger data matrix should result in 36 rows.
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({4, 4}, IndexType::get(&context));
  int64_t padding = 2;
  SmallVector<int64_t> strides = {1, 1};
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = convFilterRelation.getConstantBound64(
      BoundType::UB, convFilterRelation.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 35);
}

TEST(ConvolutionTest, ConvFilterRelationEvaluate) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  int64_t padding = 0;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  std::vector<std::vector<int>> filter = {{1, -1}, {-1, 1}};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnMatrix(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {1, -1, 0, -1, 1, 0, 0, 0, 0},
      {0, 1, -1, 0, -1, 1, 0, 0, 0},
      {0, 0, 0, 1, -1, 0, -1, 1, 0},
      {0, 0, 0, 0, 1, -1, 0, -1, 1},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, ConvFilterRelationEvaluateStrided) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({4, 4}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  std::vector<std::vector<int>> filter = {{1, 2}, {3, 4}};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnMatrix(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, ConvFilterRelationEvaluateStridedPadded) {
  MLIRContext context;
  // 3x3 data, 2x2 filter, stride 2, padding 1
  // Padded 5x5:
  // 0 0 0 0 0
  // 0 1 2 3 0
  // 0 4 5 6 0
  // 0 7 8 9 0
  // 0 0 0 0 0
  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 1;
  IntegerRelation convFilterRelation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  std::vector<std::vector<int>> filter = {{1, 2}, {3, 4}};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnMatrix(convFilterRelation, filter);

  // Output has 4 rows where the filter can slide over.
  std::vector<std::vector<int>> expected = {
      {4, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 3, 4, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 2, 0, 0, 4, 0, 0},
      {0, 0, 0, 0, 1, 2, 0, 3, 4},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, ConvChwFchwFilterRelation) {
  MLIRContext context;
  // 3x3 input and filter, with 2 input/output channels, strides = {1, 1},
  // padding = 0
  // See Figure 4 of Orion in https://arxiv.org/pdf/2311.03470.
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  int64_t padding = 1;
  IntegerRelation rel =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = rel.getConstantBound64(BoundType::UB,
                                        rel.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 17);

  auto slotBound = rel.getConstantBound64(
      BoundType::UB, rel.getVarKindOffset(VarKind::Range) + 1);
  ASSERT_TRUE(slotBound.has_value());
  EXPECT_EQ(slotBound.value(), 17);
}

TEST(ConvolutionTest, ConvChwFchwNoPaddingFilterRelation) {
  MLIRContext context;
  // f = 2, c = 2, h = 2, w = 2, strides = {2, 2}, padding = 0
  // data = (c, 4, 4)
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 4, 4}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  IntegerRelation rel =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = rel.getConstantBound64(BoundType::UB,
                                        rel.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  // singleRowSize = ((4-2)/2 + 1) * ((4-2)/2 + 1) = 2 * 2 = 4
  // f=2 -> ctBound = 1 * 4 + 3 = 7
  EXPECT_EQ(ctBound.value(), 7);

  auto slotBound = rel.getConstantBound64(
      BoundType::UB, rel.getVarKindOffset(VarKind::Range) + 1);
  ASSERT_TRUE(slotBound.has_value());
  // singleColSize = 16, c=2.
  // singleColMax = (slidingRow * 2 + filterRow) * 4 + (slidingCol * 2 +
  // filterCol) = (1 * 2 + 1) * 4 + (1 * 2 + 1) = 15. slotBound = 1 * 16 + 15
  // = 31.
  EXPECT_EQ(slotBound.value(), 31);
}

TEST(ConvolutionTest, ConvChwFchwFilterRelationUnequalStrides) {
  MLIRContext context;
  // f = 2, c = 2, h = 3, w = 3, strides = {2, 3}, padding = 0
  // data = (c, 5, 5)
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 5, 5}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 3};
  int64_t padding = 0;
  IntegerRelation rel =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = rel.getConstantBound64(BoundType::UB,
                                        rel.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  // singleRowSize = ((5-3)/2 + 1) * ((5-3)/3 + 1) = 2 * 1 = 2
  // f=2 -> ctBound = 1 * 2 + 1 = 3
  EXPECT_EQ(ctBound.value(), 3);

  auto slotBound = rel.getConstantBound64(
      BoundType::UB, rel.getVarKindOffset(VarKind::Range) + 1);
  ASSERT_TRUE(slotBound.has_value());
  // singleColSize = 25, c=2. However, the filter only touches data elements up
  // to index 22 in each channel because of the stride.
  // singleColMax = (slidingRow * 2 + filterRow) * 5 + (slidingCol * 3 +
  // filterCol) = (1 * 2 + 2) * 5 + (0 * 3 + 2) = 22. slotBound = 1 * 25 + 22
  // = 47.
  EXPECT_EQ(slotBound.value(), 47);
}

TEST(ConvolutionTest, ConvChwFchwFilterRelationPadding) {
  MLIRContext context;
  // f = 2, c = 2, h = 3, w = 3, strides = {2, 2}, padding = 1
  // data = (c, 3, 3)
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 3, 3}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 1;
  IntegerRelation rel =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);

  auto ctBound = rel.getConstantBound64(BoundType::UB,
                                        rel.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  // singleRowSize = ((3+2-3)/2 + 1) * ((3+2-3)/2 + 1) = 2 * 2 = 4
  // f=2 -> ctBound = 1 * 4 + 3 = 7
  EXPECT_EQ(ctBound.value(), 7);

  auto slotBound = rel.getConstantBound64(
      BoundType::UB, rel.getVarKindOffset(VarKind::Range) + 1);
  ASSERT_TRUE(slotBound.has_value());
  // singleColSize = 9, c=2 -> slotBound = 1 * 9 + 8 = 17
  EXPECT_EQ(slotBound.value(), 17);
}
TEST(ConvolutionTest, TestRowInterchange) {
  MLIRContext context;
  // c=4, h=2, w=2, g=2
  IntegerRelation rel = getRowInterchangeRelation(4, 2, 2, 2);

  std::vector<int> input = {0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> expectedPermutation = {0, 4, 1, 5, 8,  12, 9,  13,
                                          2, 6, 3, 7, 10, 14, 11, 15};
  PointPairCollector collector(1, 1);  // 1 domain dim, 1 range dim
  enumeratePoints(rel, collector);

  EXPECT_EQ(collector.points.size(), expectedPermutation.size());

  for (const auto& actualPoint : collector.points) {
    // The permutation in the relation is the expected (i -> j) mappings.
    auto startVal = actualPoint.first[0];
    auto permuteIdx = actualPoint.second[0];
    auto resultingVal = expectedPermutation[permuteIdx];
    EXPECT_EQ(startVal, resultingVal)
        << "Point not found: domain=" << actualPoint.first[0]
        << ", range=" << actualPoint.second[0];
  }
}

TEST(ConvolutionTest, TestRowInterchangeMultiChannel) {
  MLIRContext context;
  // c=18, h=2, w=2, g=3
  // Input: 2x2x18 = 72 elements. Output: 6x6x2
  IntegerRelation rel = getRowInterchangeRelation(18, 2, 2, 3);

  PointPairCollector collector(1, 1);  // 1 domain dim, 1 range dim
  enumeratePoints(rel, collector);

  EXPECT_EQ(collector.points.size(), 72);

  // expected contains all the flattened output channels in order
  std::vector<int> expected = {
      0,  4,  8,  1,  5,  9,  12, 16, 20, 13, 17, 21, 24, 28, 32, 25, 29, 33,
      2,  6,  10, 3,  7,  11, 14, 18, 22, 15, 19, 23, 26, 30, 34, 27, 31, 35,
      36, 40, 44, 37, 41, 45, 48, 52, 56, 49, 53, 57, 60, 64, 68, 61, 65, 69,
      38, 42, 46, 39, 43, 47, 50, 54, 58, 51, 55, 59, 62, 66, 70, 63, 67, 71};

  for (const auto& actualPoint : collector.points) {
    // The permutation in the relation is the expected (i -> j) mappings.
    auto startVal = actualPoint.first[0];
    auto permuteIdx = actualPoint.second[0];
    auto resultingVal = expected[permuteIdx];
    EXPECT_EQ(startVal, resultingVal)
        << "Point not found: domain=" << actualPoint.first[0]
        << ", range=" << actualPoint.second[0];
    ;
  }
}

TEST(ConvolutionTest, TestStrideTwoConvolution) {
  MLIRContext context;

  RankedTensorType filterType =
      RankedTensorType::get({2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({28, 28}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  IntegerRelation rel =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  PointPairCollector collector(
      2, 2);  // 2 domain dims (fh, fw), 2 range dims (ct, slot)
  enumeratePoints(rel, collector);

  // Check a few specific expected points: (f, c, fh, fw) -> (ct, slot)
  auto containsPoint = [&](std::vector<int64_t> domain,
                           std::vector<int64_t> range) {
    for (const auto& p : collector.points) {
      if (p.first == domain && p.second == range) return true;
    }
    return false;
  };

  // The expected output size is 196x784
  EXPECT_TRUE(containsPoint({0, 0}, {0, 0}));
  EXPECT_TRUE(containsPoint({0, 1}, {0, 1}));
  EXPECT_TRUE(containsPoint({1, 0}, {0, 28}));
  EXPECT_TRUE(containsPoint({1, 1}, {0, 29}));

  // Second row of application
  EXPECT_TRUE(containsPoint({0, 0}, {1, 2}));
  EXPECT_TRUE(containsPoint({0, 1}, {1, 3}));
  EXPECT_TRUE(containsPoint({1, 0}, {1, 30}));
  EXPECT_TRUE(containsPoint({1, 1}, {1, 31}));
}

TEST(ConvolutionTest, TestMultiChannelMultiRow) {
  // Pools a 1x4x28x28 into a 1x4x14x14
  // The filter is 4x4x2x2
  MLIRContext context;

  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 28, 28}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;

  auto rel =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);

  // Number of ciphertexts is number of elements of a single result (14*14) *
  // num channels = 196 * 4 = 784.
  auto ctOffset = rel.getVarKindOffset(VarKind::Range);
  auto ctBound = rel.getConstantBound64(BoundType::UB, ctOffset);
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 783);

  PointPairCollector collector(
      4, 2);  // 4 domain dims (f, c, fh, fw), 2 range dims (ct, slot)
  enumeratePoints(rel, collector);

  // Check a few specific expected points: (f, c, fh, fw) -> (ct, slot)
  auto containsPoint = [&](std::vector<int64_t> domain,
                           std::vector<int64_t> range) {
    for (const auto& p : collector.points) {
      if (p.first == domain && p.second == range) return true;
    }
    return false;
  };

  EXPECT_TRUE(containsPoint({0, 0, 0, 0}, {0, 0}));
  EXPECT_TRUE(containsPoint({0, 0, 0, 1}, {0, 1}));
  EXPECT_TRUE(containsPoint({0, 1, 0, 0}, {0, 784}));
}

TEST(ConvolutionTest, TestMultiChannelMultiRowDiagonalized) {
  // Pools a 1x4x28x28 into a 1x4x14x14
  // The filter is 4x4x2x2
  MLIRContext context;

  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 28, 28}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  int64_t ciphertextSize = 4096;

  auto maybeRel = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, ciphertextSize, false);
  ASSERT_TRUE(succeeded(maybeRel));
  IntegerRelation rel = maybeRel.value();

  // Number of ciphertexts is number of elements of a single result (14*14) *
  // num channels = 196 * 4 = 784. The matrix is diagonalized, so the bound is
  // the next power of two.
  auto ctOffset = rel.getVarKindOffset(VarKind::Range);
  auto ctBound = rel.getConstantBound64(BoundType::UB, ctOffset);
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 1023);

  PointPairCollector collector(
      4, 2);  // 4 domain dims (f, c, fh, fw), 2 range dims (ct, slot)
  enumeratePoints(rel, collector);

  // Check a few specific expected points: (f, c, fh, fw) -> (ct, slot)
  auto containsPoint = [&](std::vector<int64_t> domain,
                           std::vector<int64_t> range) {
    for (const auto& p : collector.points) {
      if (p.first == domain && p.second == range) return true;
    }
    return false;
  };

  EXPECT_TRUE(containsPoint({0, 0, 0, 0}, {0, 0}));
  EXPECT_TRUE(containsPoint({0, 0, 0, 1}, {1, 0}));
  EXPECT_TRUE(containsPoint({0, 1, 0, 0}, {784, 0}));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
