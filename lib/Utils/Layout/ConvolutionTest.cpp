#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/ConvolutionTestUtil.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
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

TEST(ConvolutionTest, Conv1dFilterRelationEvaluate) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({5}, IndexType::get(&context));
  int64_t stride = 1;
  int64_t padding = 1;
  IntegerRelation convFilterRelation =
      get1dConvFilterRelation(filterType, dataType, stride, padding);

  std::vector<int> filter = {1, 2, 3};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnVector(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {2, 3, 0, 0, 0}, {1, 2, 3, 0, 0}, {0, 1, 2, 3, 0},
      {0, 0, 1, 2, 3}, {0, 0, 0, 1, 2},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, Conv1dFilterRelationEvaluateWithPadding) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3}, IndexType::get(&context));
  int64_t stride = 1;
  int64_t padding = 2;
  IntegerRelation convFilterRelation =
      get1dConvFilterRelation(filterType, dataType, stride, padding);

  std::vector<int> filter = {1, 2, 3};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnVector(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {3, 0, 0}, {2, 3, 0}, {1, 2, 3}, {0, 1, 2}, {0, 0, 1},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, Conv1dFilterRelationEvaluateEvenSizedKernel) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({4}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({5}, IndexType::get(&context));
  int64_t stride = 1;
  int64_t padding = 1;
  IntegerRelation convFilterRelation =
      get1dConvFilterRelation(filterType, dataType, stride, padding);

  std::vector<int> filter = {1, 2, 3, 4};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnVector(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {2, 3, 4, 0, 0},
      {1, 2, 3, 4, 0},
      {0, 1, 2, 3, 4},
      {0, 0, 1, 2, 3},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, Conv1dFilterRelationEvaluateStridedEvenSizedKernel) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({6}, IndexType::get(&context));
  int64_t stride = 3;
  int64_t padding = 0;
  IntegerRelation convFilterRelation =
      get1dConvFilterRelation(filterType, dataType, stride, padding);

  std::vector<int> filter = {1, 2};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnVector(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {1, 2, 0, 0, 0},
      {0, 0, 0, 1, 2},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, Conv1dFilterRelationEvaluateWithStride) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({6}, IndexType::get(&context));
  int64_t stride = 2;
  int64_t padding = 1;
  IntegerRelation convFilterRelation =
      get1dConvFilterRelation(filterType, dataType, stride, padding);

  std::vector<int> filter = {1, 2, 3};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnVector(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {2, 3, 0, 0, 0, 0},
      {0, 1, 2, 3, 0, 0},
      {0, 0, 0, 1, 2, 3},
  };
  EXPECT_EQ(packedFilter, expected);
}

TEST(ConvolutionTest, Conv1dFilterRelationEvaluateExpanded) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({6}, IndexType::get(&context));
  int64_t stride = 1;
  int64_t padding = 1;
  IntegerRelation convFilterRelation =
      get1dConvFilterRelation(filterType, dataType, stride, padding);

  std::vector<int> filter = {1, 2, 3};
  std::vector<std::vector<int>> packedFilter =
      evaluateLayoutOnVector(convFilterRelation, filter);

  std::vector<std::vector<int>> expected = {
      {2, 3, 0, 0, 0, 0}, {1, 2, 3, 0, 0, 0}, {0, 1, 2, 3, 0, 0},
      {0, 0, 1, 2, 3, 0}, {0, 0, 0, 1, 2, 3}, {0, 0, 0, 0, 1, 2},
  };

  RankedTensorType expanded =
      get1dConvFilterExpandedType(filterType, dataType, stride, padding);

  EXPECT_EQ(packedFilter, expected);
  EXPECT_EQ(expanded.getDimSize(0), expected.size());
  EXPECT_EQ(expanded.getDimSize(1), expected[0].size());
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

TEST(ConvolutionTest, Conv1DCwFcwFilterRelation) {
  MLIRContext context;
  // length 5 input, length 3 kernel and filter, with 2 input/output channels,
  // stride = 1, padding = 1
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 5}, IndexType::get(&context));
  int64_t stride = 1;
  int64_t padding = 1;
  IntegerRelation rel =
      get1dConvCwFcwFilterRelation(filterType, dataType, stride, padding);

  // One filter contributes Datasize-Kernelsize+1 +2 *padding = 5-3+1 +2 = 5
  // rows. Two filters means the upper bound is 2*3-1 = 5
  auto ctBound = rel.getConstantBound64(BoundType::UB,
                                        rel.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 9);

  // datasize is 5. 2 input channels means the bound is 2*5-1 = 9
  auto slotBound = rel.getConstantBound64(
      BoundType::UB, rel.getVarKindOffset(VarKind::Range) + 1);
  ASSERT_TRUE(slotBound.has_value());
  EXPECT_EQ(slotBound.value(), 9);
}

TEST(ConvolutionTest, Conv1DCwFcwNoPaddingFilterRelation) {
  MLIRContext context;
  // data length = 5, kernel length=2, stride =2 => output of dimension 2
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 4}, IndexType::get(&context));
  int64_t stride = 2;
  int64_t padding = 0;
  IntegerRelation rel =
      get1dConvCwFcwFilterRelation(filterType, dataType, stride, padding);

  auto ctBound = rel.getConstantBound64(BoundType::UB,
                                        rel.getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  // RowSize =2, f=2 => ctBound = 2 + 1 = 3
  EXPECT_EQ(ctBound.value(), 3);

  auto slotBound = rel.getConstantBound64(
      BoundType::UB, rel.getVarKindOffset(VarKind::Range) + 1);
  ASSERT_TRUE(slotBound.has_value());
  EXPECT_EQ(slotBound.value(), 7);
}

TEST(ConvolutionTest, Conv2DChwFchwNoPaddingFilterRelation) {
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

TEST(ConvolutionTest, Conv2DChwFchwFilterRelationUnequalStrides) {
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

TEST(ConvolutionTest, Conv2DChwFchwFilterRelationPadding) {
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
  IntegerRelation rel = get2dConvRowInterchangeRelation(4, 2, 2, 2);

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

TEST(ConvolutionTest, Test1dConvRowInterchange) {
  MLIRContext context;
  // c=4, h=2, w=2, g=2
  IntegerRelation rel = get1dConvRowInterchangeRelation(4, 4, 2);

  std::vector<int> input = {0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> expectedPermutation = {0, 4,  1, 5,  2,  6,  3,  7,
                                          8, 12, 9, 13, 10, 14, 11, 15};
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
  IntegerRelation rel = get2dConvRowInterchangeRelation(18, 2, 2, 3);

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

TEST(ConvolutionTest, TestRowInterchangeNonSquareFilter) {
  MLIRContext context;
  // Models filter 4x1x2x3=NCHW (non-square) on input 1x1x4x7=FCHW with stride
  // 2:
  //   outputH = (4-2)/2 + 1 = 2
  //   outputW = (7-3)/2 + 1 = 3
  // f=4 (output channels), h=2, w=3, g=2 (stride).
  // Input flat size = f*h*w = 24.
  IntegerRelation rel = get2dConvRowInterchangeRelation(4, 2, 3, 2);

  PointPairCollector collector(1, 1);  // 1 domain dim, 1 range dim
  enumeratePoints(rel, collector);

  // The relation should be a 1-to-1 permutation covering all 24 input indices.
  EXPECT_EQ(collector.points.size(), 24);

  // As a sanity check, take c=0, hi=1, wi=2.
  // This maps to idx_in = 5
  // [(wi + hi * outputW + outputH*OutputW*c) = 2 + 1*3 + 0]
  // The idx_out is then, with co = 0, h0 = 2, w0 = 4
  // idx_out = 4 + 2 * wOut + 0 * wOut * hOut = 16
  EXPECT_TRUE(rel.containsPointNoLocal({5, 16}));
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

TEST(ConvolutionTest, TestConv1dMultiChannelMultiRow) {
  // Pools a 1x4x28 into a 1x4x14
  // The filter is 4x4x2
  MLIRContext context;

  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 28}, IndexType::get(&context));
  int64_t stride = 2;
  int64_t padding = 0;

  auto rel =
      get1dConvCwFcwFilterRelation(filterType, dataType, stride, padding);

  // Number of ciphertexts is number of elements of a single result (14) *
  // num channels = 14 * 4 = 56.
  auto ctOffset = rel.getVarKindOffset(VarKind::Range);
  auto ctBound = rel.getConstantBound64(BoundType::UB, ctOffset);
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 55);

  PointPairCollector collector(
      3, 2);  // 3 domain dims (f, c,  fw), 2 range dims (ct, slot)
  enumeratePoints(rel, collector);

  // Check a few specific expected points: (f, c, fw) -> (ct, slot)
  auto containsPoint = [&](std::vector<int64_t> domain,
                           std::vector<int64_t> range) {
    for (const auto& p : collector.points) {
      if (p.first == domain && p.second == range) return true;
    }
    return false;
  };

  EXPECT_TRUE(containsPoint({0, 0, 0}, {0, 0}));
  EXPECT_TRUE(containsPoint({0, 0, 1}, {0, 1}));
  EXPECT_TRUE(containsPoint({0, 1, 0}, {0, 28}));
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
  int64_t minSlotCount = 4096;

  auto maybeRel = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, minSlotCount, false);
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

TEST(ConvolutionTest, TestMultiChannelMultiRowDiagonalizedInterchanged) {
  MLIRContext context;

  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 28, 28}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  int64_t minSlotCount = 4096;

  auto maybeRel = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, minSlotCount, true);
  ASSERT_TRUE(succeeded(maybeRel));
  IntegerRelation rel = maybeRel.value();

  PointPairCollector collector(
      4, 2);  // 4 domain dims (f, c, fh, fw), 2 range dims (ct, slot)
  enumeratePoints(rel, collector);

  std::set<int64_t> nonzeroCiphertexts;
  for (const auto& p : collector.points) {
    int64_t f = p.first[0];
    int64_t c = p.first[1];
    // For pooling, filter is only non-zero when f == c
    if (f == c) {
      nonzeroCiphertexts.insert(p.second[0]);  // p.second[0] is ct
    }
  }

  EXPECT_EQ(nonzeroCiphertexts.size(), 16);

  std::set<int64_t> totalCiphertexts;
  for (const auto& p : collector.points) {
    totalCiphertexts.insert(p.second[0]);
  }
  EXPECT_EQ(totalCiphertexts.size(), 36);
}

TEST(ConvolutionTest, TestConv1dCwFcwDiagonalizedRowInterchange) {
  // Without the interchange, we first iterate over the kernel positions and hit
  // all diagonals With the interchange, we first iterate over the filters, and
  // hit only 3 diagonals
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({2, 1, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 1, 8}, IndexType::get(&context));
  int64_t stride = 2;
  int64_t padding = 0;
  int64_t minSlotCount = 8;

  auto distinctDiagonals = [&](bool interchangeRows) {
    auto maybeRel = get1dConvCwFcwFilterDiagonalizedRelation(
        filterType, dataType, stride, padding, minSlotCount, interchangeRows);
    EXPECT_TRUE(succeeded(maybeRel));
    PointPairCollector collector(/*domainDims=*/3, /*rangeDims=*/2);
    enumeratePoints(maybeRel.value(), collector);
    std::set<int64_t> cts;
    for (const auto& p : collector.points) cts.insert(p.second[0]);
    return cts;
  };

  auto withoutInterchange = distinctDiagonals(/*interchangeRows=*/false);
  EXPECT_EQ(withoutInterchange.size(), 8);

  auto withInterchange = distinctDiagonals(/*interchangeRows=*/true);
  EXPECT_EQ(withInterchange.size(), 3);
}

// Reference dense expanded Toeplitz matrix for a 1-D multichannel convolution
// with the given stride and a symmetric zero padding of `padding` on the width
// dim. Rows are (f, ow) row-major; columns index the *unpadded* data as (c, w)
// row-major, so a window position that reaches into the padding simply
// contributes no column.
std::vector<std::vector<int>> reference1dConvCwFcwMatrix(
    const std::vector<std::vector<std::vector<int>>>& filter, int64_t dataWidth,
    int64_t stride, int64_t padding) {
  int64_t outputChannels = filter.size();
  int64_t inputChannels = filter[0].size();
  int64_t filterWidth = filter[0][0].size();
  int64_t outputWidth = (dataWidth + 2 * padding - filterWidth) / stride + 1;

  std::vector<std::vector<int>> matrix(
      outputChannels * outputWidth,
      std::vector<int>(inputChannels * dataWidth, 0));
  for (int64_t f = 0; f < outputChannels; ++f) {
    for (int64_t ow = 0; ow < outputWidth; ++ow) {
      for (int64_t c = 0; c < inputChannels; ++c) {
        for (int64_t fw = 0; fw < filterWidth; ++fw) {
          int64_t w = ow * stride - padding + fw;
          if (w < 0 || w >= dataWidth) continue;
          matrix[f * outputWidth + ow][c * dataWidth + w] = filter[f][c][fw];
        }
      }
    }
  }
  return matrix;
}

// Inverts the squat-diagonal packing applied by diagonalize2dMatrix, which maps
// (row, col) -> (ct, slot) via `slot % paddedRows == row` and
// `(ct + slot) % paddedCols == col`. Returns a paddedRows x paddedCols dense
// matrix so that entries landing outside the logical matrix are visible rather
// than dropped.
std::vector<std::vector<int>> undiagonalizeMatrix(
    const std::vector<std::vector<int>>& packed, int64_t rows, int64_t cols) {
  int64_t paddedRows = (int64_t)nextPowerOfTwo(rows);
  int64_t paddedCols = (int64_t)nextPowerOfTwo(cols);
  std::vector<std::vector<int>> dense(paddedRows,
                                      std::vector<int>(paddedCols, 0));
  for (int64_t ct = 0; ct < (int64_t)packed.size(); ++ct) {
    for (int64_t slot = 0; slot < (int64_t)packed[ct].size(); ++slot) {
      dense[slot % paddedRows][(ct + slot) % paddedCols] = packed[ct][slot];
    }
  }
  return dense;
}

// Pads `matrix` out to paddedRows x paddedCols with zeros.
std::vector<std::vector<int>> padMatrixToPowerOfTwo(
    const std::vector<std::vector<int>>& matrix) {
  int64_t paddedRows = (int64_t)nextPowerOfTwo(matrix.size());
  int64_t paddedCols = (int64_t)nextPowerOfTwo(matrix[0].size());
  std::vector<std::vector<int>> result(paddedRows,
                                       std::vector<int>(paddedCols, 0));
  for (size_t i = 0; i < matrix.size(); ++i) {
    for (size_t j = 0; j < matrix[i].size(); ++j) {
      result[i][j] = matrix[i][j];
    }
  }
  return result;
}

// Checks that get1dConvCwFcwFilterDiagonalizedRelation encodes exactly the
// reference Toeplitz matrix for the given conv parameters.
void checkConv1dCwFcwDiagonalized(MLIRContext& context, int64_t outputChannels,
                                  int64_t inputChannels, int64_t filterWidth,
                                  int64_t dataWidth, int64_t stride,
                                  int64_t padding, int64_t ciphertextSize) {
  SCOPED_TRACE("f=" + std::to_string(outputChannels) +
               " c=" + std::to_string(inputChannels) + " k=" +
               std::to_string(filterWidth) + " w=" + std::to_string(dataWidth) +
               " stride=" + std::to_string(stride) +
               " padding=" + std::to_string(padding));

  // Deterministic non-zero filter values, so that an entry landing in the wrong
  // row or column is visible rather than coincidentally zero.
  std::vector<std::vector<std::vector<int>>> filter(
      outputChannels, std::vector<std::vector<int>>(
                          inputChannels, std::vector<int>(filterWidth, 0)));
  for (int64_t f = 0; f < outputChannels; ++f) {
    for (int64_t c = 0; c < inputChannels; ++c) {
      for (int64_t k = 0; k < filterWidth; ++k) {
        filter[f][c][k] = (int)((f * 37 + c * 11 + k * 3) % 17) + 1;
      }
    }
  }
  std::function<int(const std::vector<int64_t>&)> getFilterValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]];
  };

  RankedTensorType filterType = RankedTensorType::get(
      {outputChannels, inputChannels, filterWidth}, IndexType::get(&context));
  RankedTensorType dataType = RankedTensorType::get(
      {1, inputChannels, dataWidth}, IndexType::get(&context));

  auto expandedType =
      get1dConvCwFcwFilterExpandedType(filterType, dataType, stride, padding);
  auto expected =
      reference1dConvCwFcwMatrix(filter, dataWidth, stride, padding);
  int64_t rows = expandedType.getDimSize(0);
  int64_t cols = expandedType.getDimSize(1);
  ASSERT_EQ(rows, (int64_t)expected.size());
  ASSERT_EQ(cols, (int64_t)expected[0].size());

  // The non-diagonalized relation must agree with the reference Toeplitz
  // matrix. evaluateLayout maps (matRow, matCol) directly onto (ct, slot) here,
  // so the result is the dense matrix -- sized explicitly, since the relation's
  // own derived bounds are tighter than the expanded type whenever a trailing
  // data column is never touched by any window.
  auto expandedRelation =
      get1dConvCwFcwFilterRelation(filterType, dataType, stride, padding);
  EXPECT_EQ(evaluateLayout(expandedRelation, getFilterValueFn,
                           SmallVector<int64_t>{rows, cols}),
            expected);

  // diagonalize2dMatrix reads the matrix shape off those derived bounds, while
  // the Halevi-Shoup kernel is handed get1dConvCwFcwFilterExpandedType. The two
  // must agree once rounded up to a power of two, or the packed diagonals are
  // interpreted against the wrong row/column stride.
  auto rowBound = expandedRelation.getConstantBound64(
      BoundType::UB, expandedRelation.getVarKindOffset(VarKind::Range));
  auto colBound = expandedRelation.getConstantBound64(
      BoundType::UB, expandedRelation.getVarKindOffset(VarKind::Range) + 1);
  ASSERT_TRUE(rowBound.has_value() && colBound.has_value());
  EXPECT_EQ(nextPowerOfTwo(rowBound.value() + 1), nextPowerOfTwo(rows));
  EXPECT_EQ(nextPowerOfTwo(colBound.value() + 1), nextPowerOfTwo(cols));

  // ... and so must the diagonalized relation that production actually uses.
  auto maybeRel = get1dConvCwFcwFilterDiagonalizedRelation(
      filterType, dataType, stride, padding, ciphertextSize,
      /*interchangeRows=*/false);
  ASSERT_TRUE(succeeded(maybeRel));
  auto packed = evaluateLayout(maybeRel.value(), getFilterValueFn);
  EXPECT_EQ(undiagonalizeMatrix(packed, rows, cols),
            padMatrixToPowerOfTwo(expected));
}

TEST(ConvolutionTest, TestConv1dCwFcwDiagonalizedStride2WithPadding) {
  // padding == 0 is included as a control: it validates the reference matrix
  // and the un-diagonalization, so a failure only at padding > 0 isolates the
  // bug to the padded strided path.
  MLIRContext context;
  for (int64_t padding : {0, 1, 2}) {
    checkConv1dCwFcwDiagonalized(context, /*outputChannels=*/2,
                                 /*inputChannels=*/2, /*filterWidth=*/3,
                                 /*dataWidth=*/6, /*stride=*/2, padding,
                                 /*ciphertextSize=*/16);
  }
}

TEST(ConvolutionTest, TestConv1dCwFcwDiagonalizedPaddingExceedsStride) {
  // Padding larger than the stride, so the leading windows are mostly
  // padding and no window starts at data index 0.
  MLIRContext context;
  checkConv1dCwFcwDiagonalized(context, /*outputChannels=*/24,
                               /*inputChannels=*/16, /*filterWidth=*/9,
                               /*dataWidth=*/48, /*stride=*/2, /*padding=*/4,
                               /*ciphertextSize=*/1024);
}

// Checks that the filter layout LayoutPropagation assigns to a 2-D conv
// encodes exactly the reference Toeplitz matrix for the given conv parameters.
void checkConv2dChwFchwDiagonalized(
    MLIRContext& context, int64_t outputChannels, int64_t inputChannels,
    int64_t filterSize, int64_t dataH, int64_t dataW, int64_t stride,
    int64_t padding, int64_t ciphertextSize, bool interchangeRows) {
  SCOPED_TRACE("f=" + std::to_string(outputChannels) +
               " c=" + std::to_string(inputChannels) +
               " k=" + std::to_string(filterSize) +
               " h=" + std::to_string(dataH) + " w=" + std::to_string(dataW) +
               " stride=" + std::to_string(stride) +
               " padding=" + std::to_string(padding) +
               " interchangeRows=" + std::to_string(interchangeRows));

  ConvTensor4D filter = deterministicConvFilter(outputChannels, inputChannels,
                                                filterSize, filterSize);
  std::function<int(const std::vector<int64_t>&)> getFilterValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]]
                 [domainPoint[3]];
  };

  RankedTensorType filterType = RankedTensorType::get(
      {outputChannels, inputChannels, filterSize, filterSize},
      IndexType::get(&context));
  RankedTensorType dataType = RankedTensorType::get(
      {1, inputChannels, dataH, dataW}, IndexType::get(&context));
  SmallVector<int64_t> strides = {stride, stride};

  auto expandedType = get2dConvChwFchwFilterExpandedType(filterType, dataType,
                                                         padding, strides);
  auto expected =
      reference2dConvChwFchwMatrix(filter, dataH, dataW, stride, padding);
  int64_t rows = expandedType.getDimSize(0);
  int64_t cols = expandedType.getDimSize(1);
  ASSERT_EQ(rows, (int64_t)expected.size());
  ASSERT_EQ(cols, (int64_t)expected[0].size());

  // The non-diagonalized relation must agree with the reference Toeplitz
  // matrix.
  auto expandedRelation =
      get2dConvChwFchwFilterRelation(filterType, dataType, strides, padding);
  EXPECT_EQ(evaluateLayout(expandedRelation, getFilterValueFn,
                           SmallVector<int64_t>{rows, cols}),
            expected);

  // Row interchange permutes the matrix rows into the pixel-shuffled order the
  // gapped output layout uses, i.e. the order get2dConvRowInterchangeRelation
  // assigns: (f, oh, ow) row-major over (outputChannels, outputH, outputW)
  // becomes (f / g^2, oh * g + (f % g^2) / g, ow * g + f % g) row-major over
  // (outputChannels / g^2, outputH * g, outputW * g).
  std::vector<std::vector<int>> expectedRows = expected;
  if (interchangeRows) {
    int64_t g = stride;
    int64_t outputH = convOutputExtent(dataH, filterSize, stride, padding);
    int64_t outputW = convOutputExtent(dataW, filterSize, stride, padding);
    int64_t wOut = outputW * g;
    ASSERT_EQ(outputChannels % (g * g), 0);
    for (int64_t f = 0; f < outputChannels; ++f) {
      for (int64_t oh = 0; oh < outputH; ++oh) {
        for (int64_t ow = 0; ow < outputW; ++ow) {
          int64_t from = (f * outputH + oh) * outputW + ow;
          int64_t to =
              ((f / (g * g) * (outputH * g) + oh * g + (f % (g * g)) / g) *
               wOut) +
              ow * g + f % g;
          expectedRows[to] = expected[from];
        }
      }
    }
  }

  auto maybeRels = get2dConvChwFchwFilterAsSequence(
      filterType, dataType, strides, padding, ciphertextSize, interchangeRows);
  ASSERT_TRUE(succeeded(maybeRels));
  IntegerRelation composed = maybeRels->front();
  for (const auto& rel : llvm::drop_begin(maybeRels.value())) {
    composed.compose(rel);
  }
  auto packed = evaluateLayout(composed, getFilterValueFn);
  EXPECT_EQ(undiagonalizeMatrix(packed, rows, cols),
            padMatrixToPowerOfTwo(expectedRows));
}

TEST(ConvolutionTest, TestConv2dChwFchwDiagonalizedStride2WithPadding) {
  // padding == 0 is included as a control: it validates the reference matrix
  // and the un-diagonalization, so a failure only at padding > 0 isolates the
  // bug to the padded strided path.
  MLIRContext context;
  for (int64_t padding : {0, 1, 2}) {
    checkConv2dChwFchwDiagonalized(context, /*outputChannels=*/2,
                                   /*inputChannels=*/2, /*filterSize=*/3,
                                   /*dataH=*/6, /*dataW=*/6, /*stride=*/2,
                                   padding, /*ciphertextSize=*/128,
                                   /*interchangeRows=*/false);
  }
}

TEST(ConvolutionTest, TestConv2dChwFchwDiagonalizedPaddingExceedsStride) {
  // Padding larger than the stride, so the leading windows are mostly
  // padding and no window starts at data index (0, 0).
  MLIRContext context;
  checkConv2dChwFchwDiagonalized(context, /*outputChannels=*/4,
                                 /*inputChannels=*/4, /*filterSize=*/3,
                                 /*dataH=*/6, /*dataW=*/6, /*stride=*/2,
                                 /*padding=*/3, /*ciphertextSize=*/256,
                                 /*interchangeRows=*/false);
}

TEST(ConvolutionTest, TestConv2dChwFchwDiagonalizedSamePadding) {
  // The stride-1 "same" convolution that LayoutPropagation folds a tensor.pad
  // into: the output keeps the data's spatial extents.
  MLIRContext context;
  checkConv2dChwFchwDiagonalized(context, /*outputChannels=*/4,
                                 /*inputChannels=*/4, /*filterSize=*/3,
                                 /*dataH=*/4, /*dataW=*/4, /*stride=*/1,
                                 /*padding=*/1, /*ciphertextSize=*/64,
                                 /*interchangeRows=*/false);
}

TEST(ConvolutionTest, TestConv2dChwFchwDiagonalizedInterchangedPadded) {
  // LayoutPropagation turns row interchange on exactly when the stride exceeds
  // 1, so this is the combination a strided conv with a folded tensor.pad
  // selects. padding == 0 is the control.
  MLIRContext context;
  for (int64_t padding : {0, 1}) {
    checkConv2dChwFchwDiagonalized(context, /*outputChannels=*/4,
                                   /*inputChannels=*/2, /*filterSize=*/3,
                                   /*dataH=*/6, /*dataW=*/6, /*stride=*/2,
                                   padding, /*ciphertextSize=*/128,
                                   /*interchangeRows=*/true);
  }
}

TEST(ConvolutionTest, TestConv2dChwFchwDiagonalizedInterchangedNonSquare) {
  // A non-square spatial output, where transposing the two row-flattening
  // extents in the interchanged path silently drops matrix rows. Every other
  // conv test here is square, which cannot tell the two extents apart.
  MLIRContext context;
  for (int64_t padding : {0, 1}) {
    checkConv2dChwFchwDiagonalized(context, /*outputChannels=*/4,
                                   /*inputChannels=*/2, /*filterSize=*/3,
                                   /*dataH=*/6, /*dataW=*/8, /*stride=*/2,
                                   padding, /*ciphertextSize=*/128,
                                   /*interchangeRows=*/true);
  }
}

// A gap-2 packing of a (1, 2, 4) operand: element (c, w) sits in slot
// 2 * (4c + w). `low` shifts the layout's spatial domain, which is what
// tensor.pad does to the layout of the value it pads.
IntegerRelation gappedDataLayout(int64_t low) {
  std::string lowStr = std::to_string(low);
  return getIntegerRelationFromIslStr(
             "{ [n, c, w] -> [ct, slot] : n = 0 and ct = 0 and 0 <= c <= 1 "
             "and " +
             lowStr + " <= w <= 3 + " + lowStr + " and slot = 2 * (4c + w - " +
             lowStr + ") }")
      .value();
}

TEST(ConvolutionTest, TestConv1dDataColumnPermutation) {
  MLIRContext context;
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 4}, IndexType::get(&context));

  auto permutation = get1dConvDataColumnPermutation(
      dataType, gappedDataLayout(/*low=*/0), /*padding=*/0);
  ASSERT_TRUE(succeeded(permutation));

  // Column j reads slot 2j: the matrix consumes the gapped packing in place.
  std::vector<std::pair<int64_t, int64_t>> expected;
  for (int64_t j = 0; j < 8; ++j) expected.push_back({j, 2 * j});
  EXPECT_EQ(collectSlots(permutation.value()), expected);
}

TEST(ConvolutionTest, TestConv1dDataColumnPermutationFoldedPadding) {
  // A pad of 1 folded into the conv's own padding: the matrix is built against
  // the unpadded (1, 2, 4) operand, while the layout indexes the padded
  // (1, 2, 6) value. Column j must still read the slot holding unpadded
  // element j, so the same gap-2 packing comes back as column j -> slot 2j.
  MLIRContext context;
  RankedTensorType matrixDataType =
      RankedTensorType::get({1, 2, 4}, IndexType::get(&context));

  auto permutation = get1dConvDataColumnPermutation(
      matrixDataType, gappedDataLayout(/*low=*/1), /*padding=*/1);
  ASSERT_TRUE(succeeded(permutation));

  std::vector<std::pair<int64_t, int64_t>> expected;
  for (int64_t j = 0; j < 8; ++j) expected.push_back({j, 2 * j});
  EXPECT_EQ(collectSlots(permutation.value()), expected);
}

TEST(ConvolutionTest, TestConv1dDataColumnPermutationRejectsInteriorHole) {
  // Same folded-pad shape as above, but the layout maps only every third
  // spatial index, so columns 1, 2, 5 and 6 get no slot. The lowest and highest
  // columns still have one, so the bounds run 0..7 exactly as in the healthy
  // case, and a check that only compares them accepts this and silently drops
  // four real elements from the plaintext matrix. With a fold every column is
  // real data, so it must fail.
  MLIRContext context;
  RankedTensorType matrixDataType =
      RankedTensorType::get({1, 2, 4}, IndexType::get(&context));

  // Without a fold the columns still span the padded value, so a column with no
  // slot may be a pad zero and the caller decides. This case shows the bounds
  // cannot see the hole: columns 0 and 7 are both mapped, yet half are missing.
  IntegerRelation unfoldedHole =
      getIntegerRelationFromIslStr(
          "{ [n, c, w] -> [ct, slot] : n = 0 and ct = 0 and 0 <= c <= 1 and "
          "0 <= w <= 3 and w mod 3 = 0 and slot = 2 * (4c + w) }")
          .value();
  auto unchecked = get1dConvDataColumnPermutation(matrixDataType, unfoldedHole,
                                                  /*padding=*/0);
  ASSERT_TRUE(succeeded(unchecked));
  auto mapped = getMappedConvMatrixColumns(unchecked.value());
  EXPECT_EQ(mapped.size(), 4u);
  EXPECT_TRUE(mapped.contains(0));
  EXPECT_TRUE(mapped.contains(7));

  // With a fold every column is real data, so the same shape of hole fails.
  IntegerRelation foldedHole =
      getIntegerRelationFromIslStr(
          "{ [n, c, w] -> [ct, slot] : n = 0 and ct = 0 and 0 <= c <= 1 and "
          "1 <= w <= 4 and (w - 1) mod 3 = 0 and slot = 2 * (4c + w - 1) }")
          .value();
  EXPECT_TRUE(failed(get1dConvDataColumnPermutation(matrixDataType, foldedHole,
                                                    /*padding=*/1)));
}

TEST(ConvolutionTest, TestConv1dCwFcwDiagonalizedAbsorbsGappedPacking) {
  // The end of the chain the two changes above serve: a folded pad and a gapped
  // packing at once. The matrix is built against the unpadded (1, 2, 4) operand
  // with the pad of 1 in its own `padding` parameter, and it absorbs the gap-2
  // packing of the padded value. The result must be the reference Toeplitz
  // matrix with column j moved to slot 2j, and zero everywhere else.
  MLIRContext context;
  const int64_t outputChannels = 2;
  const int64_t inputChannels = 2;
  const int64_t filterWidth = 3;
  const int64_t dataWidth = 4;
  const int64_t stride = 1;
  const int64_t padding = 1;
  const int64_t ciphertextSize = 16;

  std::vector<std::vector<std::vector<int>>> filter(
      outputChannels, std::vector<std::vector<int>>(
                          inputChannels, std::vector<int>(filterWidth, 0)));
  for (int64_t f = 0; f < outputChannels; ++f) {
    for (int64_t c = 0; c < inputChannels; ++c) {
      for (int64_t k = 0; k < filterWidth; ++k) {
        filter[f][c][k] = (int)((f * 37 + c * 11 + k * 3) % 17) + 1;
      }
    }
  }
  std::function<int(const std::vector<int64_t>&)> getFilterValueFn =
      [&](const std::vector<int64_t>& domainPoint) -> int {
    return filter[domainPoint[0]][domainPoint[1]][domainPoint[2]];
  };

  RankedTensorType filterType = RankedTensorType::get(
      {outputChannels, inputChannels, filterWidth}, IndexType::get(&context));
  RankedTensorType matrixDataType = RankedTensorType::get(
      {1, inputChannels, dataWidth}, IndexType::get(&context));

  auto permutation = get1dConvDataColumnPermutation(
      matrixDataType, gappedDataLayout(/*low=*/padding), padding);
  ASSERT_TRUE(succeeded(permutation));

  auto maybeRel = get1dConvCwFcwFilterDiagonalizedRelation(
      filterType, matrixDataType, stride, padding, ciphertextSize,
      /*interchangeRows=*/false, &permutation.value());
  ASSERT_TRUE(succeeded(maybeRel));

  auto expected =
      reference1dConvCwFcwMatrix(filter, dataWidth, stride, padding);
  int64_t rows = expected.size();
  std::vector<std::vector<int>> expectedGapped(
      rows, std::vector<int>(ciphertextSize, 0));
  for (int64_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < expected[row].size(); ++col) {
      expectedGapped[row][2 * col] = expected[row][col];
    }
  }

  auto packed = evaluateLayout(maybeRel.value(), getFilterValueFn);
  EXPECT_EQ(undiagonalizeMatrix(packed, rows, ciphertextSize), expectedGapped);
}

TEST(ConvolutionTest, TestConv1dDataColumnPermutationRejectsWrongPadding) {
  // The layout was shifted by 1, so reading it back with a padding of 2 moves
  // the window past the end of its domain and leaves columns with no slot.
  // Dropping those columns would drop real data, so this must fail instead.
  MLIRContext context;
  RankedTensorType matrixDataType =
      RankedTensorType::get({1, 2, 4}, IndexType::get(&context));

  EXPECT_TRUE(failed(get1dConvDataColumnPermutation(
      matrixDataType, gappedDataLayout(/*low=*/1), /*padding=*/2)));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
