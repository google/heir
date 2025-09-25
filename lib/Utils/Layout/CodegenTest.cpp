#include <cstdint>
#include <string>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

// ISL

namespace mlir {
namespace heir {
namespace {

using ::testing::Eq;

TEST(CodegenTest, PureAffineEquality) {
  MLIRContext context;
  auto relation = getIntegerRelationFromIslStr(
      "{ [d0] -> [d1] : d0 - d1 = 0 and d0 >= 0 and d1 >= 0 and 10 >= d0 and "
      "10 "
      ">= d1 }");
  ASSERT_TRUE(succeeded(relation));
  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 10; c0 += 1)
  S(c0, c0);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, EqualityWithMod) {
  MLIRContext context;
  auto relation = getIntegerRelationFromIslStr(
      "{ [d0] -> [d1] : (d0 - d1) mod 2 = 0 and d0 >= 0 and d1 >= 0 and 10 >= "
      "d0 and 30 >= d1 }");
  ASSERT_TRUE(succeeded(relation));

  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 30; c0 += 1)
  for (int c1 = -((c0 + 1) % 2) + 1; c1 <= 10; c1 += 2)
    S(c1, c0);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, HaleviShoup) {
  MLIRContext context;
  // Data is 32x64, being packed into ciphertexts of size 1024 via Halevi-Shoup
  // diagonal layout.
  auto relation = getIntegerRelationFromIslStr(
      "{ [row, col] -> [ct, slot] : (slot - row) mod 32 = 0 and (ct + slot - "
      "col) mod 64 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 "
      "and 1023 >= slot and 31 >= ct and 31 >= row and 63 >= col }");
  ASSERT_TRUE(succeeded(relation));

  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 31; c0 += 1)
  for (int c1 = 0; c1 <= 1023; c1 += 1)
    S(c1 % 32, -((-c0 - c1 + 1087) % 64) + 63, c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, HaleviShoupWithSimplify) {
  MLIRContext context;
  auto relation = getIntegerRelationFromIslStr(
      "{ [row, col] -> [ct, slot] : "
      "(slot - row) mod 512 = 0 and "
      "(ct + slot - col) mod 512 = 0 and "
      "row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and "
      "1023 >= slot and 511 >= ct and 511 >= row and 511 >= col }");
  ASSERT_TRUE(succeeded(relation));

  // A test to ensure simplifying the relation in FPL doesn't break codegen,
  // which it did in an earlier iteration of the codegen routine.
  relation.value().simplify();
  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 511; c0 += 1)
  for (int c1 = 0; c1 <= 1023; c1 += 1)
    S(c1 % 512, (c0 + c1) % 512, c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, RowMajor) {
  MLIRContext context;
  // Data is 32 being packed into ciphertexts of size 1024 via row-major
  // layout.
  auto relation = getIntegerRelationFromIslStr(
      "{ [row] -> [ct, slot] : (slot - row) mod 32 = 0 and row >= 0 and ct >= "
      "0 and slot >= 0 and 1023 >= slot and 0 >= ct and 31 >= row }");
  ASSERT_TRUE(succeeded(relation));
  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c1 = 0; c1 <= 1023; c1 += 1)
  S(c1 % 32, 0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, HaleviShoupSquat) {
  MLIRContext context;
  auto relation = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 16 = 0 and (-i0 + slot) "
      "mod 16 = 0 and 0 <= i0 <= 9 and 0 <= i1 <= 15 and 0 <= ct <= 15 and 0 "
      "<= slot <= 1023 }");
  ASSERT_TRUE(succeeded(relation));

  // Generated code has if statements
  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 15; c0 += 1)
  for (int c1 = 0; c1 <= 1023; c1 += 1)
    if ((c1 + 6) % 16 >= 6)
      S(c1 % 16, (c0 + c1) % 16, c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, HaleviShoupSquatVecmat) {
  MLIRContext context;
  // This is a layout produced for a vecmat with a matrix of size 5x3.
  auto relation = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : (-i0 + i1 + ct) mod 4 = 0 and (-i0 + ct + "
      "slot) mod 8 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 2 and 0 <= ct <= 3 and "
      "0 <= slot <= 7 }");
  ASSERT_TRUE(succeeded(relation));

  // Generated code has if statements
  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 3; c0 += 1)
  for (int c1 = 0; c1 <= 6; c1 += 1)
    if ((c0 + c1 + 3) % 8 >= 3 && c1 % 4 <= 2)
      S((c0 + c1) % 8, c1 % 4, c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, ConvFilterRelationGenerated) {
  // This is a layout produced for a convolution kernel with 3x3 filter and
  // padding = strides = 1.
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  int64_t padding = 1;
  auto relation = get2dConvFilterRelation(filterType, dataType, padding);

  auto result = generateLoopNestAsCStr(relation);
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 8; c0 += 1)
  for (int c1 = max(max(max(-1, c0 - 4), -(c0 % 3) + c0 - 3), -(c0 % 3)); c1 <= min(min(min(9, c0 + 4), -(c0 % 3) + c0 + 5), -(c0 % 3) + 10); c1 += 1)
    if (c1 + 1 >= (-c0 + c1 + 4) % 3 && ((-c0 + c1 + 4) % 3) + 7 >= c1 && ((-c0 + c1 + 4) % 3) + (c0 % 3) >= 1 && ((-c0 + c1 + 4) % 3) + (c0 % 3) <= 3)
      S((-c0 + c1 + 4) / 3, (-c0 + c1 + 4) % 3, c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, ConvFilterRelationNoPadding) {
  // This is a layout produced for a convolution kernel with 3x3 filter and
  // padding = 0
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({3, 3}, IndexType::get(&context));
  int64_t padding = 0;
  auto relation = get2dConvFilterRelation(filterType, dataType, padding);

  auto result = generateLoopNestAsCStr(relation);
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c1 = 0; c1 <= 8; c1 += 1)
  S(c1 / 3, c1 % 3, 0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
