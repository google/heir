#include <cstdint>
#include <string>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Convolution.h"
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

TEST(CodegenTest, Conv2dScheduleComparison) {
  MLIRContext context;
  // Use a smaller size for manual verification
  RankedTensorType filterType =
      RankedTensorType::get({2, 2, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 2, 4, 4}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  int64_t ciphertextSize = 64;

  auto maybeRel = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, ciphertextSize, false);
  ASSERT_TRUE(succeeded(maybeRel));
  auto rel = maybeRel.value();

  // Schedule 1: Default (Range-only)
  auto result1 = generateLoopNestAsCStr(rel, {});
  ASSERT_TRUE(succeeded(result1));
  std::string expected1 = R"(
for (int c0 = 0; c0 <= 7; c0 += 1)
  for (int c1 = 0; c1 <= 63; c1 += 1)
    if (2 * (c1 % 2) + 4 * (c1 % 4) + c0 + 32 * floord(-(2 * (c1 % 2) + 4 * (c1 % 4)) - c0 + 7 * c1 + 1, 32) >= 7 * c1 || (7 * c1 >= 2 * (c1 % 2) + 4 * (c1 % 4) + c0 + 32 * floord(-(2 * (c1 % 2) + 4 * (c1 % 4)) - c0 + 7 * c1 + 1, 32) + 27 && 2 * (c1 % 2) + 4 * (c1 % 4) + c0 + 32 * floord(-(2 * (c1 % 2) + 4 * (c1 % 4)) - c0 + 7 * c1 + 1, 32) + 28 >= 7 * c1) || ((-c0 + c1 + 9) % 4 <= 1 && 7 * c1 >= 2 * (c1 % 2) + 4 * (c1 % 4) + c0 + 32 * floord(-(2 * (c1 % 2) + 4 * (c1 % 4)) - c0 + 7 * c1 + 1, 32) + 11 && 2 * (c1 % 2) + 4 * (c1 % 4) + c0 + 32 * floord(-(2 * (c1 % 2) + 4 * (c1 % 4)) - c0 + 7 * c1 + 1, 32) + 15 >= 7 * c1) || 2 * (c1 % 2) + 4 * (c1 % 4) + c0 + 32 * floord(-(2 * (c1 % 2) + 4 * (c1 % 4)) - c0 + 7 * c1 + 1, 32) + 16 == 7 * c1)
      S(c1 / 4 - 2 * (c1 / 8), 14 * (c1 % 2) + 7 * c0 + 7 * c1 + 3 >= 8 * ((c0 - c1 + 64) % 4) + 28 * (c1 % 4) + 224 * floord(c1 + 2 * floord(c0 - c1, 4), 16) && 8 * ((c0 - c1 + 64) % 4) + 28 * (c1 % 4) + 224 * floord(c1 + 2 * floord(c0 - c1, 4), 16) + 28 >= 14 * (c1 % 2) + 7 * c0 + 7 * c1 ? 0 : 1, 7 * c1 >= 4 * (c1 % 4) + 2 * (c1 % 2) + c0 + 32 * floord(-(2 * (c1 % 4)) + 3 * c1 - 2 * floord(c0 - c1, 4), 16) + 9 && 4 * (c1 % 4) + 2 * (c1 % 2) + c0 + 32 * floord(-(2 * (c1 % 4)) + 3 * c1 - 2 * floord(c0 - c1, 4), 16) + 16 >= 7 * c1 ? (c1 % 4) - (c1 % 8) - c1 / 2 - (-c0 + c1 + 3) / 4 + 8 * ((2 * (c1 % 8) - 2 * (c1 % 4) + c1 + 2 * ((-c0 + c1 + 3) / 4)) / 16) + 4 : (c1 % 8) - (c1 % 4) + 7 * (c1 / 2) - 7 * floord(c0 - c1, 4) - 8 * floord(14 * (c1 % 8) - (48 * (c1 % 2) + 14 * (c1 % 4)) + 49 * c1 - 98 * floord(c0 - c1, 4), 112), (c0 - c1 + 64) % 4, c0, c1);
)";
  ASSERT_THAT(result1.value(), Eq(expected1));

  // Schedule 2: Optimized (Include f and c in outer loops)
  auto result2 = generateLoopNestAsCStr(rel, {0, 1});
  ASSERT_TRUE(succeeded(result2));
  std::string expectedPrefix2 = R"(
for (int c0 = 0; c0 <= 1; c0 += 1)
  for (int c1 = 0; c1 <= 1; c1 += 1)
    for (int c2 = 0; c2 <= 7; c2 += 1)
      for (int c3 = 4 * c0; c3 <= 4 * c0 + 59; c3 += 1)
)";
  ASSERT_THAT(result2.value(), testing::StartsWith(expectedPrefix2));

  // Schedule 3: Full schedule (include all dimensions in the schedule)
  auto result3 = generateLoopNestAsCStr(rel, {0, 1, 2, 3});
  ASSERT_TRUE(succeeded(result3));
  std::string expectedPrefix3 = R"(
for (int c0 = 0; c0 <= 1; c0 += 1)
  for (int c1 = 0; c1 <= 1; c1 += 1)
    for (int c2 = 0; c2 <= 1; c2 += 1)
      for (int c3 = 0; c3 <= 1; c3 += 1)
        for (int c4 = 0; c4 <= 7; c4 += 1)
          for (int c5 = -((c3 - c4 + 7) % 4) + 4 * c0 + 3; c5 <= 4 * c0 + 59; c5 += 4)
)";
  ASSERT_THAT(result3.value(), testing::StartsWith(expectedPrefix3));
}

TEST(CodegenTest, Conv2dNoInterchangeToFromAttrTest) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();
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
  auto rel = maybeRel.value();

  // Scheduling f (idx 0) and c (idx 1) as outer loops to improve performance.
  SmallVector<int> domainIndicesToSchedule = {0, 1};

  auto result = generateLoopNestAsCStr(rel, domainIndicesToSchedule);
  ASSERT_TRUE(succeeded(result));
  std::string expected = R"(
for (int c0 = 0; c0 <= 3; c0 += 1)
  for (int c1 = 0; c1 <= 3; c1 += 1)
    for (int c2 = 0; c2 <= 1023; c2 += 1)
      for (int c3 = 196 * c0; c3 <= 4095; c3 += 1)
        if ((((-c3 + 4879) % 1024) + 196 * c0 <= 783 && ((-c3 + 4879) % 1024) + 196 * c0 >= 588 && 2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) >= 784 * c1 + 3 * c3) || (((-c3 + 4879) % 1024) + 196 * c0 <= 783 && ((-c3 + 4879) % 1024) + 196 * c0 >= 588 && 784 * c1 + 3 * c3 >= 2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) + 4067 && 2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) + 4068 >= 784 * c1 + 3 * c3))
          S(c0, c1, 196 * c0 + floord(c2 - c3, 4) + 1024 * floord(-392 * c0 + 392 * c1 + c3 - 2 * floord(c2 - c3, 4), 2048) == 196 * c1 + (c3 + 240) / 1024 + 7 * ((c3 - 2 * ((c3 + 240) / 1024)) / 14) && (140 * c0 - 140 * c1 + 147 * floord(c2 - c3, 4) - 147 * ((c3 + 240) / 1024) - 5 * ((c3 - 2 * ((c3 + 240) / 1024)) / 14)) % 1024 == 0 && (28 * c0 - 28 * c1 - 73 * floord(c2 - c3, 4) + 73 * ((c3 + 240) / 1024) - (c3 - 2 * ((c3 + 240) / 1024)) / 14) % 512 == 0 && (28 * c0 - 28 * c1 + 439 * floord(c2 - c3, 4) - 439 * ((c3 + 240) / 1024) - (c3 - 2 * ((c3 + 240) / 1024)) / 14) % 1024 == 0 && (-28 * c0 + 28 * c1 - 439 * floord(c2 - c3, 4) + 439 * ((c3 + 240) / 1024) + (c3 - 2 * ((c3 + 240) / 1024)) / 14) % 1024 == 0 ? 0 : 1, (c2 - c3 + 4096) % 4, c2, c3);
)";
  ASSERT_THAT(result.value(), Eq(expected));

  // convert to layoutattr
  auto layout = tensor_ext::LayoutAttr::getFromIntegerRelation(&context, rel);

  // now interpret the layout attr and then try to run generateLoopNest
  auto rel2 = layout.getIntegerRelation();
  result = generateLoopNestAsCStr(rel2, domainIndicesToSchedule);
  ASSERT_TRUE(succeeded(result));
  std::string expected2 = R"(
for (int c0 = 0; c0 <= 3; c0 += 1)
  for (int c1 = 0; c1 <= 3; c1 += 1)
    for (int c2 = 0; c2 <= 1023; c2 += 1)
      for (int c3 = 0; c3 <= 4095; c3 += 1)
        if (((-196 * c0 + c3 + 828) % 1024 >= 828 && 2 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) >= 784 * c1 + 3 * c3) || ((-196 * c0 + c3 + 828) % 1024 >= 828 && 784 * c1 + 3 * c3 >= 2 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) + 4067 && 2 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) + 4068 >= 784 * c1 + 3 * c3))
          S(c0, c1, 196 * c0 + floord(c2 - c3, 4) + 1024 * floord(-392 * c0 + 392 * c1 + c3 - 2 * floord(c2 - c3, 4), 2048) == 196 * c1 + (-196 * c0 + c3 + 828) / 1024 + 7 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) / 14) && (140 * c0 - 140 * c1 + 147 * floord(c2 - c3, 4) - 147 * ((-196 * c0 + c3 + 828) / 1024) - 5 * ((c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) / 14)) % 1024 == 0 && (28 * c0 - 28 * c1 - 73 * floord(c2 - c3, 4) + 73 * ((-196 * c0 + c3 + 828) / 1024) - (c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) / 14) % 512 == 0 && (28 * c0 - 28 * c1 + 439 * floord(c2 - c3, 4) - 439 * ((-196 * c0 + c3 + 828) / 1024) - (c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) / 14) % 1024 == 0 && (-28 * c0 + 28 * c1 - 439 * floord(c2 - c3, 4) + 439 * ((-196 * c0 + c3 + 828) / 1024) + (c3 - 2 * ((-196 * c0 + c3 + 828) / 1024)) / 14) % 1024 == 0 ? 0 : 1, (c2 - c3 + 4096) % 4, c2, c3);
)";
  ASSERT_THAT(result.value(), Eq(expected2));
}

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
  SmallVector<int64_t> strides = {1, 1};
  auto relation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

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
  SmallVector<int64_t> strides = {1, 1};
  auto relation =
      get2dConvFilterRelation(filterType, dataType, strides, padding);

  auto result = generateLoopNestAsCStr(relation);
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c1 = 0; c1 <= 8; c1 += 1)
  S(c1 / 3, c1 % 3, 0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, ConvFilterMIMOPermuted) {
  MLIRContext context;

  // Filter size: 4x4x2x2 (f, c, h, w)
  // Input size: 1x4x10x10 (batch, c, h, w)
  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 10, 10}, IndexType::get(&context));
  SmallVector<int64_t> strides = {1, 1};
  int64_t padding = 0;
  int64_t ciphertextSize = 1024;

  auto relation = get2dConvChwFchwFilterDiagonalizedRelation(
      filterType, dataType, strides, padding, ciphertextSize, true);
  ASSERT_TRUE(succeeded(relation));

  auto result = generateLoopNestAsCStr(relation.value());
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 15; c0 += 1)
  for (int c1 = 0; c1 <= 1023; c1 += 1)
    if ((10 * c0 + 102 >= (c0 + c1 + 184) % 512 && (c0 + c1 + 184) % 512 >= c0 + 184 && c0 + 199 >= (c0 + c1 + 184) % 512 && (c0 + c1 + 184) % 512 >= 10 * c0 + 85) || (9 * c0 + 8 >= c1 % 16 && (c1 % 16) + 9 >= 9 * c0 && (c0 + c1 + 184) % 512 >= c0 + 184 && c0 + 199 >= (c0 + c1 + 184) % 512) || (9 * ((c0 + c1 + 184) % 512) >= 10 * (c1 % 16) + 1756 && 9 * c0 + 90 * floord(-9 * c0 + c1 + 20 * (c1 / 16) + 18 * ((c0 + c1 + 184) / 512) + 9, 90) + 8 >= c1 + 20 * (c1 / 16) + 18 * ((c0 + c1 + 184) / 512) && ((-(10 * (c1 % 16)) + 9 * c0 + 9 * c1 - 108 * ((c0 + c1 + 184) / 512) + 800) % 900) + 10 * c1 >= 10 * (c1 % 16) + 160 * (c1 / 16) + 809) || (c1 + 8 >= 9 * c0 + 10 * (c1 / 16) + 192 * ((c0 + c1 + 184) / 512) && 9 * c0 + 10 * (c1 / 16) + 192 * ((c0 + c1 + 184) / 512) + 8 >= c1 && 9 * ((c0 + c1 + 184) % 512) >= 10 * (c1 % 16) + 1756 && c1 / 16 - 6 * ((-(10 * (c1 % 16)) + 9 * c0 + 9 * c1 - 108 * ((c0 + c1 + 184) / 512) + 800) / 900) == 2 * ((c0 + c1 + 184) / 512)))
      S(0, c0 + c1 / 16 - 5 * ((c0 + c1 + 184) / 512) - (891 * c0 + c1 + 740 * (c1 / 16) + 108 * ((c0 + c1 + 184) / 512) + 99) / 900, -9 * c0 - 8 * (c1 / 16) - (c0 + c1 + 184) / 512 - (81 * c0 + c1 + 20 * (c1 / 16) + 18 * ((c0 + c1 + 184) / 512) + 9) / 90 + 10 * ((891 * c0 + c1 + 740 * (c1 / 16) + 108 * ((c0 + c1 + 184) / 512) + 99) / 900), -9 * c0 - 2 * (c1 / 16) - 2 * ((c0 + c1 + 184) / 512) - (c1 + 2 * (c1 / 16)) / 9 + 10 * ((81 * c0 + c1 + 20 * (c1 / 16) + 18 * ((c0 + c1 + 184) / 512) + 9) / 90), c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, Pooling) {
  MLIRContext context;
  // pools a 1x1x4x4 into a 1x1x2x2
  // the filter is 1x1x2x2, stride 2
  auto maybeRel = get2dConvChwFchwFilterDiagonalizedRelation(
      RankedTensorType::get({1, 1, 2, 2}, IndexType::get(&context)),
      RankedTensorType::get({1, 1, 4, 4}, IndexType::get(&context)), {2, 2}, 0,
      16, /*interchangeRows=*/false);
  ASSERT_TRUE(succeeded(maybeRel));
  auto relation = maybeRel.value();

  auto result = generateLoopNestAsCStr(relation);
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 3; c0 += 1)
  for (int c1 = 0; c1 <= -c0 + 15; c1 += 1)
    if (c0 >= c1 + 4 * floord(c0 - c1 + 2, 4) && 2 * (c1 % 2) + c0 + c1 + 2 >= 4 * (c1 % 4) && 4 * (c1 % 4) + 5 >= 2 * (c1 % 2) + c0 + c1)
      S(0, 0, -(c1 % 4) + c1 - c1 / 2 - (-c0 + c1 + 3) / 4, -((-c0 + c1 + 3) % 4) + 3, c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, Conv2dNoInterchangeManySchedulesTest) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();
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
  auto rel = maybeRel.value();

  // Scheduling without added domain vars, slow to process
  // This string is  used in debug_packing_4d to validate that multiple
  // schedules produce the same packing.
  //   auto result = generateLoopNestAsCStr(rel);
  //   ASSERT_TRUE(succeeded(result));
  //   llvm::dbgs() << "Schedule 1 (no domain vars): \n";
  //   llvm::dbgs() << result.value() << "\n";

  // Scheduling f (idx 0) and c (idx 1) as outer loops to improve performance.
  auto result = generateLoopNestAsCStr(rel, {0, 1});
  ASSERT_TRUE(succeeded(result));
  auto expected = R"(
for (int c0 = 0; c0 <= 3; c0 += 1)
  for (int c1 = 0; c1 <= 3; c1 += 1)
    for (int c2 = 0; c2 <= 1023; c2 += 1)
      for (int c3 = 196 * c0; c3 <= 4095; c3 += 1)
        if ((((-c3 + 4879) % 1024) + 196 * c0 <= 783 && ((-c3 + 4879) % 1024) + 196 * c0 >= 588 && 2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) >= 784 * c1 + 3 * c3) || (((-c3 + 4879) % 1024) + 196 * c0 <= 783 && ((-c3 + 4879) % 1024) + 196 * c0 >= 588 && 784 * c1 + 3 * c3 >= 2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) + 4067 && 2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14) + 784 * c0 + c2 + 4096 * floord(-(2 * ((c3 - 2 * ((c3 + 240) / 1024)) % 14)) - 784 * c0 + 784 * c1 - c2 + 3 * c3 + 1, 4096) + 4068 >= 784 * c1 + 3 * c3))
          S(c0, c1, 196 * c0 + floord(c2 - c3, 4) + 1024 * floord(-392 * c0 + 392 * c1 + c3 - 2 * floord(c2 - c3, 4), 2048) == 196 * c1 + (c3 + 240) / 1024 + 7 * ((c3 - 2 * ((c3 + 240) / 1024)) / 14) && (140 * c0 - 140 * c1 + 147 * floord(c2 - c3, 4) - 147 * ((c3 + 240) / 1024) - 5 * ((c3 - 2 * ((c3 + 240) / 1024)) / 14)) % 1024 == 0 && (28 * c0 - 28 * c1 - 73 * floord(c2 - c3, 4) + 73 * ((c3 + 240) / 1024) - (c3 - 2 * ((c3 + 240) / 1024)) / 14) % 512 == 0 && (28 * c0 - 28 * c1 + 439 * floord(c2 - c3, 4) - 439 * ((c3 + 240) / 1024) - (c3 - 2 * ((c3 + 240) / 1024)) / 14) % 1024 == 0 && (-28 * c0 + 28 * c1 - 439 * floord(c2 - c3, 4) + 439 * ((c3 + 240) / 1024) + (c3 - 2 * ((c3 + 240) / 1024)) / 14) % 1024 == 0 ? 0 : 1, (c2 - c3 + 4096) % 4, c2, c3);
)";
  ASSERT_THAT(result.value(), Eq(expected));

  // Scheduling f, c, h, w as outer loops to improve performance.
  result = generateLoopNestAsCStr(rel, {0, 1, 2, 3});
  ASSERT_TRUE(succeeded(result));
  expected = R"(
for (int c0 = 0; c0 <= 3; c0 += 1)
  for (int c1 = 0; c1 <= 3; c1 += 1)
    for (int c2 = 0; c2 <= 1; c2 += 1)
      for (int c3 = 0; c3 <= 1; c3 += 1)
        for (int c4 = 0; c4 <= 1023; c4 += 1)
          for (int c5 = -((c3 - c4 + 1023) % 4) + 196 * c0 + 3; c5 <= 4095; c5 += 4)
            if (((-c5 + 4879) % 1024) + 196 * c0 <= 783 && ((-c5 + 4879) % 1024) + 196 * c0 >= 588 && (-(2 * ((c5 - 2 * ((c5 + 240) / 1024)) % 14)) - 784 * c0 + 784 * c1 + 28 * c2 + c3 - c4 + 3 * c5) % 4096 == 0)
              S(c0, c1, c2, c3, c4, c5);
)";
  ASSERT_THAT(result.value(), Eq(expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
