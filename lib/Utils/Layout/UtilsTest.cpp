#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::VarKind;

void runRowMajorTest(RankedTensorType tensorType, int64_t numSlots) {
  IntegerRelation result = getRowMajorLayoutRelation(tensorType, numSlots);

  // Check that the result relation requires size(tensor) / slots ciphertexts.
  auto ctIndex = result.getVarKindOffset(VarKind::Range);
  std::optional<int64_t> numCiphertexts =
      result.getConstantBound64(BoundType::UB, ctIndex);
  ASSERT_TRUE(numCiphertexts.has_value());
  EXPECT_EQ(numCiphertexts.value(),
            std::ceil(tensorType.getNumElements() / (double)numSlots) - 1);

  // Ensure that the layout is row-major.
  SmallVector<int64_t> shape = llvm::to_vector(tensorType.getShape());
  for (int64_t i = 0; i < tensorType.getNumElements(); ++i) {
    SmallVector<int64_t> indices = getIndicesFromRowMajorShape(i, shape);
    indices.push_back(static_cast<int64_t>(std::floor(i / (double)numSlots)));
    indices.push_back(i % numSlots);
    auto maybeExists = result.containsPointNoLocal(indices);
    EXPECT_TRUE(maybeExists.has_value());
  }
}

TEST(UtilsTest, TestAddModConstraint) {
  auto maybeRel =
      getIntegerRelationFromIslStr("{ [x] : x >= 0 and 100 - x >= 0 }");
  ASSERT_TRUE(succeeded(maybeRel));

  auto rel = maybeRel.value();
  unsigned result = addModConstraint(rel, {1, 0}, 32);  // x % 32
  rel.convertVarKind(VarKind::Local,
                     result - rel.getVarKindOffset(VarKind::Local),
                     rel.getNumVarKind(VarKind::Local), VarKind::Range);
  for (unsigned x = 0; x <= 100; ++x) {
    EXPECT_TRUE(rel.containsPointNoLocal({x, x % 32}));
  }
}

TEST(UtilsTest, TestTryProveUnequalByVolume_DifferingDomainVars) {
  auto rel1 =
      getIntegerRelationFromIslStr("{ [x, z] -> [y] : x = 0 and y = 0 }")
          .value();
  auto rel2 =
      getIntegerRelationFromIslStr("{ [x] -> [y] : x = 0 and y = 0 }").value();
  EXPECT_TRUE(succeeded(tryProveUnequalByVolume(rel1, rel2)));
}

TEST(UtilsTest, TestTryProveUnequalByVolume_DifferingExtents) {
  auto rel1 =
      getIntegerRelationFromIslStr("{ [x] -> [y] : 0 <= x <= 10 and y = 2*x }")
          .value();
  auto rel2 =
      getIntegerRelationFromIslStr("{ [x] -> [y] : 0 <= x <= 9 and y = 2*x }")
          .value();
  EXPECT_TRUE(succeeded(tryProveUnequalByVolume(rel1, rel2)));
}

// A reversed diagonal has the same bounding box, and so the same volume, as the
// forward one, so volume alone cannot separate them. isRelationEqual still
// must, which is what IsRelationEqualDistinguishesEqualVolumeRelations covers.
TEST(UtilsTest, TestTryProveUnequalByVolume_CannotDecideEqualVolumes) {
  auto rel1 =
      getIntegerRelationFromIslStr("{ [x] -> [y] : 0 <= x <= 3 and y = x }")
          .value();
  auto rel2 =
      getIntegerRelationFromIslStr("{ [x] -> [y] : 0 <= x <= 3 and y = 3 - x }")
          .value();
  EXPECT_FALSE(succeeded(tryProveUnequalByVolume(rel1, rel2)));
}

TEST(UtilsTest, TestTryProveUnequalByVolume_SameRelation) {
  auto rel1 =
      getIntegerRelationFromIslStr("{ [x] -> [y] : 0 <= x <= 10 and y = 2*x }")
          .value();
  auto rel2 =
      getIntegerRelationFromIslStr(
          "{ [x] -> [y] : 0 <= x <= 10 and 0 <= y <= 20 and x = y / 2 }")
          .value();
  EXPECT_FALSE(succeeded(tryProveUnequalByVolume(rel1, rel2)));
}

TEST(UtilsTest, SingleCiphertext) {
  // Add row major layout relation when number of slots is exactly the number of
  // elements.
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2}, IndexType::get(&context));
  int64_t numSlots = tensorType.getNumElements();

  runRowMajorTest(tensorType, numSlots);
}

TEST(UtilsTest, TwoCiphertexts) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({4}, IndexType::get(&context));
  int64_t numSlots = 2;
  runRowMajorTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDim) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 8;
  runRowMajorTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDimSingleCiphertext) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 24;
  runRowMajorTest(tensorType, numSlots);
}

TEST(UtilsTest, DiagonalLayout) {
  MLIRContext context;

  // Diagonalize a 4x8 matrix into a 4x64 matrix.
  int64_t minSlotCount = 64;
  RankedTensorType matrixType =
      RankedTensorType::get({4, 8}, IndexType::get(&context));
  IntegerRelation diagonalRelation =
      getDiagonalLayoutRelation(matrixType, minSlotCount);

  diagonalRelation.simplify();
  for (unsigned int i = 0; i < 4; ++i) {
    for (unsigned int j = 0; j < 64; ++j) {
      auto maybeExists =
          diagonalRelation.containsPointNoLocal({j % 4, (i + j) % 8, i, j});
      EXPECT_TRUE(maybeExists.has_value());
    }
  }
}

TEST(UtilsTest, SquatDiagonalLayout) {
  MLIRContext context;

  // Diagonalize a 3x5 matrix - this will require padding the row to 4 and the
  // cols to 8
  //
  //  1  2  3  4  5  *  *  *
  //  6  7  8  9 10  *  *  *
  // 11 12 13 14 15  *  *  *
  //  *  *  *  *  *  *  *  *

  // 1  7 13  * 5 *  * *
  // 2  8 14  * * *  * *
  // 3  9 15  * * * 11 *
  // 4 10  *  * * 6 12 *
  int64_t minSlotCount = 8;
  RankedTensorType matrixType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation diagonalRelation =
      getDiagonalLayoutRelation(matrixType, minSlotCount);
  int64_t paddedRows = 4;
  int64_t paddedCols = 8;

  for (unsigned int i = 0; i < 4; ++i) {
    for (unsigned int j = 0; j < 8; ++j) {
      auto row = j % paddedRows;
      auto col = (i + j) % paddedCols;
      if (row >= matrixType.getDimSize(0) || col >= matrixType.getDimSize(1)) {
        EXPECT_FALSE(diagonalRelation.containsPointNoLocal({row, col, i, j})
                         .has_value());
      } else {
        auto maybeExists =
            diagonalRelation.containsPointNoLocal({row, col, i, j});
        EXPECT_TRUE(maybeExists.has_value());
      }
    }
  }
}

TEST(UtilsTest, BicyclicLayout3x5) {
  MLIRContext context;
  int64_t numSlots = 15;
  RankedTensorType matrixType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation bicyclicRelation =
      getBicyclicLayoutRelation(matrixType, numSlots);

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  std::vector<std::vector<int>> packedMatrix =
      evaluateLayoutOnMatrix(bicyclicRelation, matrix);

  std::vector<std::vector<int>> expected = {
      {1, 7, 13, 4, 10, 11, 2, 8, 14, 5, 6, 12, 3, 9, 15}};
  EXPECT_EQ(packedMatrix, expected);
}

TEST(UtilsTest, BicyclicLayout3x5Repeated) {
  MLIRContext context;

  int64_t numSlots = 32;
  RankedTensorType matrixType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation bicyclicRelation =
      getBicyclicLayoutRelation(matrixType, numSlots);

  std::vector<std::vector<int>> matrix = {
      {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  std::vector<std::vector<int>> packedMatrix =
      evaluateLayoutOnMatrix(bicyclicRelation, matrix);

  std::vector<std::vector<int>> expected = {
      {1, 7, 13, 4, 10, 11, 2, 8, 14, 5, 6, 12, 3, 9, 15,
       // Cyclically repeated to fill 32 slots
       1, 7, 13, 4, 10, 11, 2, 8, 14, 5, 6, 12, 3, 9, 15, 1, 7}};
  EXPECT_EQ(packedMatrix, expected);
}

TEST(UtilsTest, PeriodicReplicationRelation) {
  int64_t numSlots = 10;
  int64_t period = 3;
  IntegerRelation replication =
      getPeriodicReplicationRelation(/*numCiphertexts=*/1, numSlots, period);

  // Every target slot t is reached exactly from source slot t % period.
  for (int64_t t = 0; t < numSlots; ++t) {
    for (int64_t s = 0; s < period; ++s) {
      EXPECT_EQ(replication.containsPointNoLocal({0, s, 0, t}).has_value(),
                s == t % period);
    }
  }

  // Source slots outside the first period are not in the domain.
  EXPECT_FALSE(replication.containsPointNoLocal({0, period, 0, period}));
}

TEST(UtilsTest, BicyclicCtPtDiagonal3x5x7) {
  MLIRContext context;
  int64_t numSlots = 105;
  int64_t stride = 3;
  int64_t contractionDim = 0;
  RankedTensorType weightType =
      RankedTensorType::get({5, 7}, IndexType::get(&context));
  IntegerRelation relation =
      getBicyclicDiagonalRelation(weightType, contractionDim, stride, numSlots);

  // Initialize a 5x7 weight matrix
  std::vector<std::vector<int>> weight(5, std::vector<int>(7));
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 7; ++j) {
      weight[i][j] = i * 10 + j;
    }
  }

  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation, weight);

  // Expect n = 5 rows (diagonals) and numSlots = 105 cols
  EXPECT_EQ(packed.size(), 5);
  for (int c = 0; c < 5; ++c) {
    EXPECT_EQ(packed[c].size(), numSlots);
    for (int k = 0; k < numSlots; ++k) {
      // D_c[k] = W[(k + c * stride) mod n, k mod freeSize]
      // here n = 5, stride = 3, freeSize = 7
      int expectedRow = (k + c * 3) % 5;
      int expectedCol = k % 7;
      EXPECT_EQ(packed[c][k], weight[expectedRow][expectedCol]);
    }
  }
}

TEST(UtilsTest, BicyclicPtCtDiagonal3x5x7) {
  MLIRContext context;
  int64_t numSlots = 105;
  int64_t stride = 7;
  int64_t contractionDim = 1;
  RankedTensorType weightType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation relation =
      getBicyclicDiagonalRelation(weightType, contractionDim, stride, numSlots);

  // Initialize a 3x5 weight matrix
  std::vector<std::vector<int>> weight(3, std::vector<int>(5));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      weight[i][j] = i * 10 + j;
    }
  }

  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation, weight);

  // Expect n = 5 rows (diagonals) and numSlots = 105 cols
  EXPECT_EQ(packed.size(), 5);
  for (int c = 0; c < 5; ++c) {
    EXPECT_EQ(packed[c].size(), numSlots);
    for (int k = 0; k < numSlots; ++k) {
      // D_c[k] = W[k mod freeSize, (k + c * stride) mod n]
      // here n = 5 (dim 1), stride = 7, freeSize = 3 (dim 0)
      int expectedRow = k % 3;
      int expectedCol = (k + c * 7) % 5;
      EXPECT_EQ(packed[c][k], weight[expectedRow][expectedCol]);
    }
  }
}

TEST(UtilsTest, BicyclicDiagonalNonIntegralWrap) {
  MLIRContext context;
  int64_t numSlots = 32;
  int64_t stride = 7;
  int64_t contractionDim = 1;
  RankedTensorType weightType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation relation =
      getBicyclicDiagonalRelation(weightType, contractionDim, stride, numSlots);

  std::vector<std::vector<int>> weight(3, std::vector<int>(5));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      weight[i][j] = i * 10 + j;
    }
  }

  std::vector<std::vector<int>> packed =
      evaluateLayoutOnMatrix(relation, weight);

  EXPECT_EQ(packed.size(), 5);
  for (int c = 0; c < 5; ++c) {
    EXPECT_EQ(packed[c].size(), numSlots);
    for (int k = 0; k < numSlots; ++k) {
      int expectedRow = k % 3;
      int expectedCol = (k + c * 7) % 5;
      EXPECT_EQ(packed[c][k], weight[expectedRow][expectedCol]);
    }
  }
}

TEST(UtilsTest, TricyclicLayout2x5x7Structure) {
  MLIRContext context;
  // shape h=2, m=5, n=7
  int64_t h = 2;
  int64_t m = 5;
  int64_t n = 7;
  int64_t numSlots = h * m * n;

  RankedTensorType tensorType =
      RankedTensorType::get({h, m, n}, IndexType::get(&context));
  IntegerRelation tricyclicRelation =
      getTricyclicLayoutRelation(tensorType, numSlots);

  // value = (100*h_idx + 10*m_idx + n_idx)
  std::vector<std::vector<std::vector<int>>> tensor(
      h, std::vector<std::vector<int>>(m, std::vector<int>(n, 0)));
  for (int ih = 0; ih < h; ++ih) {
    for (int im = 0; im < m; ++im) {
      for (int in = 0; in < n; ++in) {
        tensor[ih][im][in] = ih * 100 + im * 10 + in;
      }
    }
  }

  auto packedMatrix = evaluateLayoutOnTensor(tricyclicRelation, tensor);

  // φ(tensor)[k] = tensor[k mod h][k mod m][k mod n]
  std::vector<int> expected;
  expected.reserve(numSlots);
  for (int64_t k = 0; k < numSlots; ++k) {
    int ih = k % h;
    int im = k % m;
    int in = k % n;
    expected.push_back(ih * 100 + im * 10 + in);
  }

  EXPECT_EQ(packedMatrix[0], expected);
}

TEST(UtilsTest, TricyclicLayout2x5x7Repeated) {
  MLIRContext context;
  // shape h=2, m=5, n=7
  int64_t h = 2;
  int64_t m = 5;
  int64_t n = 7;
  int64_t numSlots = 200;

  RankedTensorType tensorType =
      RankedTensorType::get({h, m, n}, IndexType::get(&context));
  IntegerRelation tricyclicRelation =
      getTricyclicLayoutRelation(tensorType, numSlots);

  // value = (100*h_idx + 10*m_idx + n_idx)
  std::vector<std::vector<std::vector<int>>> tensor(
      h, std::vector<std::vector<int>>(m, std::vector<int>(n, 0)));
  for (int ih = 0; ih < h; ++ih) {
    for (int im = 0; im < m; ++im) {
      for (int in = 0; in < n; ++in) {
        tensor[ih][im][in] = ih * 100 + im * 10 + in;
      }
    }
  }

  auto packedMatrix = evaluateLayoutOnTensor(tricyclicRelation, tensor);

  // φ(tensor)[k] = tensor[k mod h][k mod m][k mod n]
  std::vector<int> expected;
  expected.reserve(numSlots);
  for (int64_t k = 0; k < numSlots; ++k) {
    int ih = k % h;
    int im = k % m;
    int in = k % n;
    expected.push_back(ih * 100 + im * 10 + in);
  }

  EXPECT_EQ(packedMatrix[0], expected);
}

// A genuine tricyclic layout must still be recognized after routing the check
// through isRelationEqual.
TEST(UtilsTest, IsRelationTricyclicAcceptsGenuineLayout) {
  MLIRContext context;
  int64_t h = 2, m = 5, n = 7;
  int64_t numSlots = h * m * n;
  RankedTensorType tensorType =
      RankedTensorType::get({h, m, n}, IndexType::get(&context));

  EXPECT_TRUE(isRelationTricyclic(
      tensorType, numSlots, getTricyclicLayoutRelation(tensorType, numSlots)));
}

// The relation passed here IS the 1x5x7 tricyclic relation, so this returns
// true without the unit-dim guard.
TEST(UtilsTest, IsRelationTricyclicRejectsUnitDim) {
  MLIRContext context;
  int64_t h = 1, m = 5, n = 7;
  int64_t numSlots = h * m * n;
  RankedTensorType tensorType =
      RankedTensorType::get({h, m, n}, IndexType::get(&context));

  EXPECT_FALSE(isRelationTricyclic(
      tensorType, numSlots, getTricyclicLayoutRelation(tensorType, numSlots)));
}

// Same degeneracy for the rank-2 CRT layout: gcd(1, cols) == 1 lets a unit-row
// matrix through the coprimality filter.
TEST(UtilsTest, IsRelationBicyclicRejectsUnitDim) {
  MLIRContext context;
  int64_t rows = 1, cols = 7;
  int64_t numSlots = rows * cols;
  RankedTensorType matrixType =
      RankedTensorType::get({rows, cols}, IndexType::get(&context));

  EXPECT_FALSE(isRelationBicyclic(
      matrixType, numSlots, getBicyclicLayoutRelation(matrixType, numSlots)));
}

// The pair from TCResNet8's first tensor.collapse_shape (1x40x101 -> 40x101 at
// logN=13), equal but differing in representation so isObviouslyEqual cannot
// settle it. Also covered as the CollapseEqual pair in
// benchmark/isl:relation_equality_benchmark.
TEST(UtilsTest, IsRelationEqualDecidesCollapsedGapStructuredConvLayout) {
  MLIRContext context;
  auto sourceRel = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and "
      "(-101i1 - i2 + slot) mod 4096 = 0 and 0 <= i1 <= 39 and "
      "0 <= i2 <= 8191 - 101i1 and i2 <= 100 and 0 <= slot <= 8191 and "
      "8192*floor((4096 + 101i1 + i2)/8192) <= 101i1 + i2 }");
  auto resultRel = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : ct = 0 and "
      "(-101i0 - i1 + slot) mod 4096 = 0 and 0 <= i0 <= 39 and "
      "0 <= i1 <= 100 and 0 <= slot <= 8191 and "
      "8192*floor((4096 + 101i0 + i1)/8192) <= 101i0 + i1 }");
  ASSERT_TRUE(succeeded(sourceRel));
  ASSERT_TRUE(succeeded(resultRel));

  RankedTensorType sourceType =
      RankedTensorType::get({1, 40, 101}, IndexType::get(&context));
  SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
  IntegerRelation collapsed =
      collapseDimensions(sourceRel.value(), sourceType, reassociation);

  EXPECT_TRUE(isRelationEqual(collapsed, resultRel.value()));
}

// One bound changed, so the check is not just answering "true" for every gap
// structured layout it is handed. Settled by tryProveUnequalByVolume, on the
// bounding-box volume.
TEST(UtilsTest, IsRelationEqualDistinguishesGapStructuredConvLayouts) {
  auto rel1 = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and "
      "(-101i1 - i2 + slot) mod 4096 = 0 and 0 <= i1 <= 39 and "
      "0 <= i2 <= 8191 - 101i1 and i2 <= 100 and 0 <= slot <= 8191 and "
      "8192*floor((4096 + 101i1 + i2)/8192) <= 101i1 + i2 }");
  auto rel2 = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and "
      "(-101i1 - i2 + slot) mod 4096 = 0 and 0 <= i1 <= 38 and "
      "0 <= i2 <= 8191 - 101i1 and i2 <= 100 and 0 <= slot <= 8191 and "
      "8192*floor((4096 + 101i1 + i2)/8192) <= 101i1 + i2 }");
  ASSERT_TRUE(succeeded(rel1));
  ASSERT_TRUE(succeeded(rel2));

  EXPECT_FALSE(isRelationEqual(rel1.value(), rel2.value()));
}

TEST(UtilsTest, IsRelationEqualDecidesNestedFloorLayout) {
  const char* nestedFloor =
      "{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and 0 <= i1 <= 23 and "
      "4 <= i2 <= 54 and 0 <= slot <= 8191 and "
      "2048*floor((824 + slot)/2048) <= slot and "
      "2*floor((-47 + 51i1 + i2 + 51slot + 8*floor((3 - 51i1 - i2)/2048))/102) "
      "<= -19 + slot - 40*floor((3 - 51i1 - i2)/2048) and "
      "102*floor((-47 + 51i1 + i2 + 51slot + 8*floor((3 - 51i1 - i2)/2048))"
      "/102) <= -98 + 51i1 + i2 + 51slot + 8*floor((3 - 51i1 - i2)/2048) and "
      "-1947 - 102i1 - 2i2 + slot - 2048*floor((824 + slot)/2048) "
      "- 2056*floor((3 - 51i1 - i2)/2048) "
      "+ 102*floor((-47 + 51i1 + i2 + 51slot + 8*floor((3 - 51i1 - i2)/2048))"
      "/102) <= 102*floor((slot)/2) <= "
      "-1946 - 102i1 - 2i2 + slot - 2048*floor((824 + slot)/2048) "
      "- 2056*floor((3 - 51i1 - i2)/2048) "
      "+ 102*floor((-47 + 51i1 + i2 + 51slot + 8*floor((3 - 51i1 - i2)/2048))"
      "/102) }";
  auto rel = getIntegerRelationFromIslStr(nestedFloor);
  ASSERT_TRUE(succeeded(rel));

  // Restating an already-implied bound leaves the point set alone but changes
  // the constraint list, so isObviouslyEqual can no longer settle the pair and
  // the check has to reach a real decision procedure.
  IntegerRelation restated = rel.value();
  restated.addBound(BoundType::LB,
                    restated.getVarKindOffset(VarKind::Range) + 1, 0);
  ASSERT_FALSE(rel.value().isObviouslyEqual(restated));

  EXPECT_TRUE(isRelationEqual(rel.value(), restated));
}

TEST(UtilsTest, TestGetRangePoints) {
  MLIRContext context;
  auto rel = getIntegerRelationFromIslStr(
      "{ [x] : x >= 0 and 7 >= x and x mod 3 = 0 }");
  ASSERT_TRUE(succeeded(rel));
  std::vector<std::vector<int64_t>> expected = {{0}, {3}, {6}};
  PointCollector collector;
  getRangePoints(rel.value(), collector);
  EXPECT_EQ(collector.points, expected);
}

TEST(UtilsTest, TestEnumeratePoints) {
  MLIRContext context;
  // Create a relation with 1 domain variable (x) and 1 range variable (y)
  IntegerRelation rel =
      getIntegerRelationFromIslStr(
          "{ [x] -> [y] : x >= 0 and 2 >= x and y >= 0 and 1 >= y }")
          .value();
  PointPairCollector collector(1, 1);  // 1 domain dim, 1 range dim
  enumeratePoints(rel, collector);

  // Expected points: domain x range pairs for x in [0,2] and y in [0,1]
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> expected =
      {{{0}, {0}}, {{0}, {1}}, {{1}, {0}}, {{1}, {1}}, {{2}, {0}}, {{2}, {1}}};

  EXPECT_EQ(collector.points.size(), expected.size());
  for (const auto& expectedPoint : expected) {
    bool found = false;
    for (const auto& actualPoint : collector.points) {
      if (actualPoint.first == expectedPoint.first &&
          actualPoint.second == expectedPoint.second) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Expected point not found: domain="
                       << expectedPoint.first[0]
                       << ", range=" << expectedPoint.second[0];
  }
}

TEST(UtilsTest, PerRowLayout) {
  MLIRContext context;

  // Per row layout 3x5 matrix
  //  1  2  3  4  5
  //  6  7  8  9 10
  // 11 12 13 14 15
  // to
  //  1  2  3  4  5 * * *  1  2  3  4  5 * * *
  //  6  7  8  9 10 * * *  6  7  8  9 10 * * *
  // 11 12 13 14 15 * * * 11 12 13 14 15 * * *
  int64_t minSlotCount = 16;
  RankedTensorType matrixType =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation perRowRelation =
      getPerRowLayoutRelation(matrixType, minSlotCount);
  int64_t paddedCols = 8;

  for (unsigned int i = 0; i < 3; ++i) {
    for (unsigned int j = 0; j < 16; ++j) {
      auto row = i;
      auto col = j % paddedCols;
      if (col >= matrixType.getDimSize(1)) {
        EXPECT_FALSE(
            perRowRelation.containsPointNoLocal({row, col, i, j}).has_value());
      } else {
        auto maybeExists =
            perRowRelation.containsPointNoLocal({row, col, i, j});
        EXPECT_TRUE(maybeExists.has_value());
      }
    }
  }
}

TEST(UtilsTest, TestAnyRangePoint) {
  MLIRContext context;
  auto rel = getIntegerRelationFromIslStr(
      "{ [x] : x >= 0 and 7 >= x and x mod 3 = 0 }");
  ASSERT_TRUE(succeeded(rel));
  std::vector<int64_t> actual = anyRangePoint(rel.value());
  EXPECT_TRUE(rel.value().containsPointNoLocal(actual).has_value());
}

TEST(UtilsTest, TestGetCollapsedRelation) {
  MLIRContext context;
  // Collapse a 2x3x4 matrix to a 6x4 matrix.
  RankedTensorType sourceType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  RankedTensorType destType =
      RankedTensorType::get({6, 4}, IndexType::get(&context));
  SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
  IntegerRelation collapsedRelation =
      getCollapsedRelation(sourceType, destType, reassociation);

  // Evaluate layout presumes a 2-d (ct, slot) output so we can hack-ishly use
  // it here for the 2D output.
  std::vector<std::vector<std::vector<int>>> input = {{
                                                          {1, 2, 3, 4},
                                                          {5, 6, 7, 8},
                                                          {9, 10, 11, 12},
                                                      },
                                                      {
                                                          {9, 10, 11, 12},
                                                          {13, 14, 15, 16},
                                                          {17, 18, 19, 20},
                                                      }};
  std::function<int(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) {
        return input[domainPoint[0]][domainPoint[1]][domainPoint[2]];
      };

  std::vector<std::vector<int>> actual =
      evaluateLayout(collapsedRelation, getValueFn);
  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4},    {5, 6, 7, 8},     {9, 10, 11, 12},
      {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20},
  };
  EXPECT_EQ(actual, expected);
}

TEST(UtilsTest, TestGetCollapsedRelationUnitDims) {
  MLIRContext context;
  // Collapse a 1x3x4 matrix to a 3x4 matrix.
  RankedTensorType sourceType =
      RankedTensorType::get({1, 3, 4}, IndexType::get(&context));
  RankedTensorType destType =
      RankedTensorType::get({3, 4}, IndexType::get(&context));
  SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
  IntegerRelation collapsedRelation =
      getCollapsedRelation(sourceType, destType, reassociation);

  // Evaluate layout presumes a 2-d (ct, slot) output so we can hack-ishly use
  // it here for the 2D output.
  std::vector<std::vector<std::vector<int>>> input = {{
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
  }};
  std::function<int(const std::vector<int64_t>&)> getValueFn =
      [&](const std::vector<int64_t>& domainPoint) {
        return input[domainPoint[0]][domainPoint[1]][domainPoint[2]];
      };

  std::vector<std::vector<int>> actual =
      evaluateLayout(collapsedRelation, getValueFn);
  std::vector<std::vector<int>> expected = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
  };
  EXPECT_EQ(actual, expected);
}

TEST(UtilsTest, TestExpandDimensionsFromRankZero) {
  // tensor.expand_shape from tensor<f32> to tensor<1x1xf32> uses
  // an empty reassociation array.
  MLIRContext context;
  auto rel = getIntegerRelationFromIslStr(
                 "{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }")
                 .value();
  RankedTensorType resultType =
      RankedTensorType::get({1, 1}, IndexType::get(&context));
  SmallVector<ReassociationIndices> reassociation = {};
  IntegerRelation expanded = expandDimensions(rel, resultType, reassociation);

  EXPECT_EQ(expanded.getNumDomainVars(), 2u);
  EXPECT_TRUE(expanded.containsPointNoLocal({0, 0, 0, 0}));
}

TEST(UtilsTest, TestCollapseDimensionsMultipleUnitDimsInGroup) {
  MLIRContext context;
  auto rel =
      getIntegerRelationFromIslStr(
          "{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and i2 = 0 and ct = 0 and "
          "(-i1 + slot) mod 4 = 0 and 0 <= i1 <= 2 and 0 <= slot <= 1023 }")
          .value();
  RankedTensorType sourceType =
      RankedTensorType::get({1, 3, 1}, IndexType::get(&context));
  SmallVector<ReassociationIndices> reassociation = {{0, 1, 2}};
  IntegerRelation collapsed =
      collapseDimensions(rel, sourceType, reassociation);

  EXPECT_EQ(collapsed.getNumDomainVars(), 1);
  EXPECT_EQ(collapsed.getNumRangeVars(), 2);
  auto expected =
      getIntegerRelationFromIslStr(
          "{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and "
          "0 <= i0 <= 2 and 0 <= slot <= 1023 }")
          .value();
  EXPECT_TRUE(collapsed.isEqual(expected));
}

TEST(UtilsTest, TestGetSliceInsertionRelation) {
  MLIRContext context;
  // Insert a 3x4 slice into a 2x1x3x4 matrix at (1, 0, 0, 0).
  RankedTensorType sliceType =
      RankedTensorType::get({3, 4}, IndexType::get(&context));
  RankedTensorType destType =
      RankedTensorType::get({2, 1, 3, 4}, IndexType::get(&context));
  SmallVector<int64_t> offsets = {1, 0, 0, 0};
  SmallVector<int64_t> sizes = {1, 1, 3, 4};
  SmallVector<int64_t> strides = {1, 1, 1, 1};

  auto sliceRelation =
      getSliceInsertionRelation(sliceType, destType, offsets, sizes, strides);
  ASSERT_TRUE(succeeded(sliceRelation));

  // Expect two ciphertexts.
  auto ctBound = sliceRelation.value().getConstantBound64(
      BoundType::UB, sliceRelation.value().getVarKindOffset(VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 1);

  // Test the first point.
  std::vector<std::vector<int64_t>> expectedPoints = {
      {0, 0, 1, 0, 0, 0}, {0, 1, 1, 0, 0, 1}, {1, 0, 1, 0, 1, 0},
      {1, 1, 1, 0, 1, 1}, {2, 2, 1, 0, 2, 2},
  };
  for (const auto& point : expectedPoints) {
    auto maybeExists = sliceRelation.value().containsPointNoLocal(point);
    EXPECT_TRUE(maybeExists.has_value());
  }
}

TEST(UtilsTest, TestShiftVar) {
  MLIRContext context;
  auto rel =
      getIntegerRelationFromIslStr(
          "{ [x, y] -> [z] : x >= 0 and y >= 0 and z >= 0 and x + y = z }")
          .value();
  // shift x by 10. x is at pos 0.
  auto shiftedRel = shiftVar(rel, 0, 10);
  // x' = x+10, so x = x'-10
  // We check if (x'=10, y=0, z=0) is in the relation.
  EXPECT_TRUE(shiftedRel.containsPointNoLocal({10, 0, 0}).has_value());
  // We check if (x'=11, y=1, z=2) is in the relation.
  EXPECT_TRUE(shiftedRel.containsPointNoLocal({11, 1, 2}).has_value());
  EXPECT_FALSE(shiftedRel.containsPointNoLocal({1, 1, 2}).has_value());
}

TEST(UtilsTest, TestShiftVarRangeOffset) {
  MLIRContext context;
  auto rel =
      getIntegerRelationFromIslStr(
          "{ [x] -> [y, z] : x >= 0 and y >= 0 and z >= 0 and x + y = z }")
          .value();
  // shift z by 10. z is at pos 0
  auto rangeOffset = rel.getVarKindOffset(VarKind::Range);
  auto shiftedRel = shiftVar(rel, rangeOffset + 1, 10);
  // z' = z+10
  // We check if (x'=0, y=0, z=10) is in the relation.
  EXPECT_TRUE(shiftedRel.containsPointNoLocal({0, 0, 10}).has_value());
  EXPECT_TRUE(shiftedRel.containsPointNoLocal({1, 1, 12}).has_value());
  EXPECT_TRUE(shiftedRel.containsPointNoLocal({8, 1, 19}).has_value());
}

TEST(UtilsTest, TestGetSliceExtractionRelation) {
  MLIRContext context;
  // Extract a 3x4 slice from a 2x1x3x4 matrix at (1, 0, 0, 0).
  RankedTensorType sourceType =
      RankedTensorType::get({2, 1, 3, 4}, IndexType::get(&context));
  RankedTensorType sliceType =
      RankedTensorType::get({3, 4}, IndexType::get(&context));
  SmallVector<int64_t> offsets = {1, 0, 0, 0};
  SmallVector<int64_t> sizes = {1, 1, 3, 4};
  SmallVector<int64_t> strides = {1, 1, 1, 1};

  auto sliceRelation = getSliceExtractionRelation(sourceType, sliceType,
                                                  offsets, sizes, strides);
  ASSERT_TRUE(succeeded(sliceRelation));

  // Test a few points.
  // The relation maps from source indices to slice indices.
  // For example, source (1,0,0,0) maps to slice (0,0)
  std::vector<std::vector<int64_t>> expectedPoints = {
      {1, 0, 0, 0, 0, 0}, {1, 0, 0, 1, 0, 1}, {1, 0, 1, 0, 1, 0},
      {1, 0, 1, 1, 1, 1}, {1, 0, 2, 2, 2, 2},
  };
  for (const auto& point : expectedPoints) {
    auto maybeExists = sliceRelation.value().containsPointNoLocal(point);
    EXPECT_TRUE(maybeExists.has_value());
  }
}

TEST(UtilsTest, TestGetCtComplementPoints) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({8, 1024}, IndexType::get(&context));
  auto rel = getIntegerRelationFromIslStr(
      "{ [x] -> [y, slot] : x >= 0 and 7 >= y and y >= 0 and x = y and x mod 2 "
      "= 0 and 7 >= x and slot >= 0 and slot <= 1023 }");
  ASSERT_TRUE(succeeded(rel));
  std::vector<std::vector<int64_t>> expected = {{1}, {3}, {5}, {7}};
  PointCollector collector;
  getCtComplementPoints(rel.value(), collector, type);
  EXPECT_EQ(collector.points, expected);
}

TEST(UtilsTest, TestGetCtComplementFromConvRelation) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({1024, 1024}, IndexType::get(&context));
  auto rel = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : (-32i0 - i1 + ct - 4*floor((slot)/28)) mod "
      "1024 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 4 and 0 <= ct <= 1023 and slot "
      ">= 0 and -28i0 <= slot <= 895 - 28i0 and slot <= 783 and -32i0 - i1 - "
      "slot <= 4*floor((slot)/28) <= 1023 - 32i0 - i1 - slot and "
      "28*floor((slot)/28) >= -31 + i1 + slot and 28*floor((slot)/28) <= i1 + "
      "slot }");
  ASSERT_TRUE(succeeded(rel));

  // Expect that 241 to 1023 are the complement points.
  PointCollector collector;
  getCtComplementPoints(rel.value(), collector, type);

  EXPECT_EQ(collector.points.size(), 783);
  for (const auto& point : collector.points) {
    EXPECT_EQ(point.size(), 1);
    EXPECT_GE(point[0], 241);
    EXPECT_LE(point[0], 1023);
  }
}

TEST(UtilsTest, TestGetCtComplementPoolingLayer) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({2048, 8192}, IndexType::get(&context));
  auto rel = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2, i3] -> [ct, slot] : exists (e0, e1, e2, e3, e4, e5: "
      "2048e4 = -i0 - 784i1 - 28i2 - i3 + ct + 2e0 - 56e1 - 2e2 + 28e3 and "
      "8192e5 = -784i1 - 28i2 - i3 + ct + slot - 56e1 - 2e2 and 0 <= i0 <= 3 "
      "and 0 <= i1 <= 5 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 2047 "
      "and 0 <= slot <= 8191 and i0 <= 2e0 <= 27 + i0 and 0 <= e1 <= 13 and 0 "
      "<= e2 <= 13 and -1 - i0 + 2e0 <= 2e2 <= -i0 + 2e0 and 0 <= e3 <= 27 and "
      "-1 + i0 + 4e1 <= 2e3 <= i0 + 4e1) }");
  ASSERT_TRUE(succeeded(rel));
  PointCollector collector;
  getCtComplementPoints(rel.value(), collector, type);
  EXPECT_EQ(collector.points.size(), 1994);
}

TEST(UtilsTest, TestIsOneToOneSingleCiphertextPacking) {
  auto permutation = getIntegerRelationFromIslStr(
                         "{ [i] -> [ct, slot] : ct = 0 and (slot - 3i) mod 8 "
                         "= 0 and 0 <= i <= 7 and 0 <= slot <= 7 }")
                         .value();
  EXPECT_TRUE(isOneToOneSingleCiphertextPacking(permutation));

  auto replicated = getIntegerRelationFromIslStr(
                        "{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 8 = "
                        "0 and 0 <= i <= 7 and 0 <= slot <= 15 }")
                        .value();
  EXPECT_FALSE(isOneToOneSingleCiphertextPacking(replicated));

  auto multipleCiphertexts = getIntegerRelationFromIslStr(
                                 "{ [i] -> [ct, slot] : i = 4ct + slot and 0 "
                                 "<= i <= 7 and 0 <= ct <= 1 and 0 <= slot <= "
                                 "3 }")
                                 .value();
  EXPECT_FALSE(isOneToOneSingleCiphertextPacking(multipleCiphertexts));
}

TEST(UtilsTest, TestFoldVectorPermutationIntoMatrixLayout) {
  // The vector's slot = 3i permutation is folded into the matrix's column
  // indexing (col -> 3col), so a diagonal matvec can consume the un-permuted
  // ciphertext directly.
  auto vectorPermutation = getIntegerRelationFromIslStr(
                               "{ [i] -> [ct, slot] : ct = 0 and (slot - 3i) "
                               "mod 8 = 0 and 0 <= i <= 7 and 0 <= slot <= 7 }")
                               .value();
  auto matrixLayout =
      getIntegerRelationFromIslStr(
          "{ [row, col] -> [ct, slot] : (row - col + ct) mod 4 = 0 and (-col + "
          "ct + slot) mod 8 = 0 and 0 <= row <= 3 and 0 <= col <= 7 and 0 <= "
          "ct <= 3 and 0 <= slot <= 7 }")
          .value();
  auto expected =
      getIntegerRelationFromIslStr(
          "{ [i0, i1] -> [ct, slot] : (i0 + i1 + ct) mod 4 = 0 and (-3i1 + ct "
          "+ slot) mod 8 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 7 and 0 <= ct <= "
          "3 and 0 <= slot <= 7 }")
          .value();

  auto folded =
      foldVectorPermutationIntoMatrixLayout(vectorPermutation, matrixLayout);
  EXPECT_TRUE(folded.isEqual(expected));
}

TEST(UtilsTest, TestIsDenseLayout_Dense) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({2, 3}, IndexType::get(&context));
  // Trivial relation that maps from 2x3 domain to 2x3 range.
  auto rel = getIntegerRelationFromIslStr(
                 "{ [i0, i1] -> [ct, slot] : 0 <= i0 <= 1 and 0 <= i1 <= 2 and "
                 "ct = i0 and slot = i1 }")
                 .value();
  EXPECT_TRUE(isDenseLayout(rel, type));
}

TEST(UtilsTest, TestIsDenseLayout_NotDense) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({2, 3}, IndexType::get(&context));
  // Relation only maps to a single row i0 = 0.
  auto rel = getIntegerRelationFromIslStr(
                 "{ [i0, i1] -> [ct, slot] : i0 = 0 and 0 <= i1 <= 2 and ct = "
                 "i0 and slot = i1 }")
                 .value();
  EXPECT_FALSE(isDenseLayout(rel, type));
}

TEST(UtilsTest, TestIsDenseLayout_TooLarge) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({2, 3}, IndexType::get(&context));
  // Domain is 2x4 and maps to 2x4 range.
  auto rel = getIntegerRelationFromIslStr(
                 "{ [i0, i1] -> [ct, slot] : 0 <= i0 <= 1 and 0 <= i1 <= 3 and "
                 "ct = i0 and slot = i1 }")
                 .value();
  EXPECT_FALSE(isDenseLayout(rel, type));
}

TEST(UtilsTest, TestIsDenseLayout_Matvec) {
  MLIRContext context;
  RankedTensorType matrixType =
      RankedTensorType::get({16, 16}, IndexType::get(&context));
  IntegerRelation matvecRelation = getDiagonalLayoutRelation(matrixType, 16);

  RankedTensorType expectedType =
      RankedTensorType::get({16, 16}, IndexType::get(&context));

  EXPECT_TRUE(isDenseLayout(matvecRelation, expectedType));
}

TEST(UtilsTest, TestRelationSize) {
  auto rel = getIntegerRelationFromIslStr(
                 "{ [i0, i1] -> [ct, slot] : 0 <= i0 <= 1 and 0 <= i1 <= 2 and "
                 "ct = i0 and slot = i1 }")
                 .value();
  // Domain is 2x3 = 6 points.
  EXPECT_EQ(relationSize(rel), 6);
}

TEST(UtilsTest, TestRelationSizeUnbounded) {
  auto rel = getIntegerRelationFromIslStr(
                 "{ [i0, i1] -> [ct, slot] : 0 <= i0 <= 1 and 0 <= i1 and "
                 "ct = i0 and slot = i1 }")
                 .value();
  // Size is unbounded
  EXPECT_EQ(relationSize(rel), -1);
}

TEST(UtilsTest, TestRelationSizeExceedsInt64) {
  auto rel = getIntegerRelationFromIslStr(
                 "{ [i0] -> [slot] : 0 <= i0 <= 9223372036854775807 and "
                 "slot = i0 }")
                 .value();
  EXPECT_EQ(relationSize(rel), -1);
}

TEST(UtilsTest, TestRelationSizeLarge) {
  auto rel =
      getIntegerRelationFromIslStr(
          "{ [i0, i1] -> [slot] : 0 <= i0 and 49999 >= i0 and 0 <= i1 and "
          "49999 >= i1 and slot = 0 }")
          .value();
  // 50000 * 50000 = 2,500,000,000 (exceeds INT_MAX)
  EXPECT_EQ(relationSize(rel), 2500000000LL);
}

TEST(UtilsTest, TestGetPaddingRelation) {
  MLIRContext context;
  RankedTensorType unpaddedType =
      RankedTensorType::get({5}, Float32Type::get(&context));
  RankedTensorType paddedType =
      RankedTensorType::get({8}, Float32Type::get(&context));

  auto rel = getPaddingRelation(paddedType, unpaddedType, {2});

  // Domain: 0 <= p <= 7
  // Range: 0 <= s <= 4
  // Constraint: p - s = 2 => s = p - 2

  // Test some points
  // Padded index 2 should map to unpadded index 0
  EXPECT_TRUE(rel.containsPointNoLocal({2, 0}).has_value());
  // Padded index 6 should map to unpadded index 4
  EXPECT_TRUE(rel.containsPointNoLocal({6, 4}).has_value());

  // Out of bounds in padded
  EXPECT_FALSE(rel.containsPointNoLocal({8, 6}).has_value());
  // Out of bounds in unpadded (even if relation holds)
  // p = 7 => s = 5, which is out of bounds for unpadded (0 <= s <= 4)
  EXPECT_FALSE(rel.containsPointNoLocal({7, 5}).has_value());
  // p = 1 => s = -1, out of bounds
  EXPECT_FALSE(rel.containsPointNoLocal({1, -1}).has_value());
}
TEST(UtilsTest, TestRelationSubset) {
  // `from`: 2x4 box
  auto from = getIntegerRelationFromIslStr(
                  "{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i0 + 4*i1 and "
                  "0 <= i0 <= 3 and 0 <= i1 <= 1 }")
                  .value();
  // `to`: 1x4 sub-box (i1 = 0)
  auto to = getIntegerRelationFromIslStr(
                "{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 "
                "<= 3 and i1 = 0 }")
                .value();
  EXPECT_TRUE(isRelationSubset(to, from));
  EXPECT_FALSE(isRelationSubset(from, to));

  // `outside`: point not in `from`
  auto outside = getIntegerRelationFromIslStr(
                     "{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i0 + 5 and "
                     "0 <= i0 <= 3 and i1 = 0 }")
                     .value();
  EXPECT_FALSE(isRelationSubset(outside, from));
}

TEST(UtilsTest, TestRelationInjective) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({3, 5}, IndexType::get(&context));
  IntegerRelation bicyclic = getBicyclicLayoutRelation(type, 1024);
  EXPECT_TRUE(isRelationInjective(bicyclic));

  auto nonInjective = getIntegerRelationFromIslStr(
                          "{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i1 and "
                          "0 <= i0 <= 1 and 0 <= i1 <= 4 }")
                          .value();
  EXPECT_FALSE(isRelationInjective(nonInjective));
}

TEST(UtilsTest, TricyclicCtPtDiagonal2x5x7) {
  MLIRContext context;
  int64_t numSlots = 105;
  int64_t ctStride = 3;
  int64_t paddedFreeDim = 7;
  int64_t contractionDim = 1;
  RankedTensorType weightType =
      RankedTensorType::get({2, 5, 7}, IndexType::get(&context));
  IntegerRelation relation = getTricyclicDiagonalRelation(
      weightType, contractionDim, ctStride, paddedFreeDim, numSlots);

  EXPECT_TRUE(relation.containsPointNoLocal({0, 0, 0, 0, 0}).has_value());
  EXPECT_TRUE(relation.containsPointNoLocal({1, 1, 1, 0, 1}).has_value());
  EXPECT_FALSE(relation.containsPointNoLocal({0, 0, 0, 0, 1}).has_value());
}

TEST(UtilsTest, TestBicyclicReduceProjection) {
  MLIRContext context;
  RankedTensorType type =
      RankedTensorType::get({33, 65}, Float32Type::get(&context));
  IntegerRelation rel = getBicyclicLayoutRelation(type, 8192);
  EXPECT_TRUE(isRelationBicyclic(type, 8192, rel));

  auto reducedExpected =
      getIntegerRelationFromIslStr(
          "{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 33 = 0 and 0 "
          "<= i0 <= 32 and 0 <= slot <= 8191 }")
          .value();
  IntegerRelation projected = rel;
  projected.projectOut(1, 1);
  EXPECT_TRUE(isRelationEqual(projected, reducedExpected));
}
}  // namespace
}  // namespace heir
}  // namespace mlir
