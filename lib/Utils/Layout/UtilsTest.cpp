#include <cmath>
#include <cstdint>
#include <optional>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Parser.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/TensorUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

void runTest(RankedTensorType tensorType, int64_t numSlots) {
  IntegerRelation result(PresburgerSpace::getRelationSpace(
      tensorType.getRank(), /*numRange=*/2, /*numSymbol=*/0,
      /*numLocals=*/0));

  // Add bounds for tensor dimensions and ciphertext slots.
  for (int i = 0; i < tensorType.getRank(); ++i) {
    result.addBound(BoundType::UB, i, tensorType.getDimSize(i) - 1);
    result.addBound(BoundType::LB, i, 0);
  }
  for (int i = result.getVarKindOffset(VarKind::Range);
       i < result.getNumRangeVars(); ++i) {
    result.addBound(BoundType::LB, i, 0);
  }
  int ctIndex = result.getVarKindEnd(VarKind::Range) - 2;
  result.addBound(BoundType::UB, ctIndex, numSlots - 1);

  // Run the test
  addRowMajorConstraint(result, tensorType, numSlots);

  // Check that the result relation requires size(tensor) / slots ciphertexts.
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

TEST(ModConstraintTest, TestAddModConstraint) {
  MLIRContext context;

  IntegerRelation rel =
      relationFromString("(x) : (x >= 0, 100 - x >= 0)", 1, &context);
  unsigned result = addModConstraint(rel, {1, 0}, 32);  // x % 32
  rel.convertVarKind(VarKind::Local,
                     result - rel.getVarKindOffset(VarKind::Local),
                     rel.getNumVarKind(VarKind::Local), VarKind::Range);
  for (unsigned x = 0; x <= 100; ++x) {
    EXPECT_TRUE(rel.containsPointNoLocal({x, x % 32}));
  }
}

TEST(UtilsTest, SingleCiphertext) {
  // Add row major layout relation when number of slots is exactly the number of
  // elements.
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2}, IndexType::get(&context));
  int64_t numSlots = tensorType.getNumElements();

  runTest(tensorType, numSlots);
}

TEST(UtilsTest, TwoCiphertexts) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({4}, IndexType::get(&context));
  int64_t numSlots = 2;
  runTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDim) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 8;
  runTest(tensorType, numSlots);
}

TEST(UtilsTest, MultiDimSingleCiphertext) {
  MLIRContext context;
  RankedTensorType tensorType =
      RankedTensorType::get({2, 3, 4}, IndexType::get(&context));
  int64_t numSlots = 24;
  runTest(tensorType, numSlots);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
