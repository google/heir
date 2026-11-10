#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Convolution.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Verifier.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

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

TEST(CodegenTest, VariadicOpTest) {
  // This exercises variadic operations in the generated ISL.
  MLIRContext context;
  context
      .loadDialect<scf::SCFDialect, arith::ArithDialect, func::FuncDialect>();

  auto relation = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2, i3] -> [o0, o1, o2, o3, o4] : o0 = i0 and o1 = i1 and o4 "
      "= 28i2 + i3 + 56o2 + 2o3 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= i2 "
      "<= 1 and 0 <= i3 <= 1 and 0 <= o2 <= 13 and 0 <= o3 <= 13 }");
  ASSERT_TRUE(succeeded(relation));
  auto rel = relation.value();

  OpBuilder builder(&context);
  auto moduleOp = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = func::FuncOp::create(builder, builder.getUnknownLoc(),
                                     "test_func", funcType);
  auto* block = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(block);

  ImplicitLocOpBuilder locBuilder(builder.getUnknownLoc(), builder);
  MLIRLoopNestGenerator generator(locBuilder);

  auto bodyBuilder = [](OpBuilder& b, Location loc, ValueRange ivs,
                        ValueRange iterArgs) { return scf::ValueVector{}; };

  auto result = generator.generateForLoop(rel, {}, bodyBuilder);
  ASSERT_TRUE(succeeded(result));

  moduleOp.erase();
}

// Regression test for a dominance error when an scf.for is nested inside the
// then-branch of an scf.if. The else-branch of such an scf.if must yield the
// iter args that flow *into* the if, not whatever the (mutable)
// currentIterArgs_ happens to point at
TEST(CodegenTest, IfWithNestedForElseYieldDominance) {
  MLIRContext context;
  context
      .loadDialect<scf::SCFDialect, arith::ArithDialect, func::FuncDialect>();

  auto relation = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2] -> [ct, slot] : (30i0 - 32i1 - i2 + ct) mod 1024 = 0 and "
      "0 <= i0 <= 63 and 0 <= i1 <= 16 and 0 <= i2 <= 2 and 0 <= ct <= 1023 "
      "and 0 <= slot <= 4095 and 2048*floor((-30 - 30i0 + slot)/2048) >= -3967 "
      "+ slot and 2048*floor((-30 - 30i0 + slot)/2048) >= -2079 - 30i0 + i2 + "
      "slot and 2048*floor((-30 - 30i0 + slot)/2048) <= -2048 - 30i0 + slot "
      "and 2048*floor((-30 - 30i0 + slot)/2048) <= -2048 - 30i0 + i2 + slot "
      "and 2048*floor((-30 - 30i0 + slot)/2048) <= -2048 + slot }");
  ASSERT_TRUE(succeeded(relation));

  OpBuilder builder(&context);
  auto moduleOp = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = func::FuncOp::create(builder, builder.getUnknownLoc(),
                                     "test_func", funcType);
  auto* block = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(block);

  ImplicitLocOpBuilder locBuilder(builder.getUnknownLoc(), builder);

  // A single iter arg threaded through the loop nest unchanged. This forces the
  // generated scf.if ops to carry a result, and therefore an else-branch yield.
  auto init = arith::ConstantIntOp::create(locBuilder, 0, 32);

  MLIRLoopNestGenerator generator(locBuilder);
  auto bodyBuilder = [](OpBuilder& b, Location loc, ValueRange ivs,
                        ValueRange iterArgs) {
    return scf::ValueVector(iterArgs.begin(), iterArgs.end());
  };

  SmallVector<int> domainIndicesToSchedule = {0, 1};
  auto result = generator.generateForLoop(relation.value(), {init.getResult()},
                                          bodyBuilder, domainIndicesToSchedule);
  ASSERT_TRUE(succeeded(result));

  func::ReturnOp::create(locBuilder, ValueRange{});

  ASSERT_TRUE(succeeded(verify(moduleOp)));

  moduleOp.erase();
}

TEST(CodegenTest, BooleanExprsStayI1) {
  MLIRContext context;
  context
      .loadDialect<scf::SCFDialect, arith::ArithDialect, func::FuncDialect>();

  // Same relation as IfWithNestedForElseYieldDominance: its floor-div
  // structure makes ISL emit `if` guards inside the loop nest.
  auto relation = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2] -> [ct, slot] : (30i0 - 32i1 - i2 + ct) mod 1024 = 0 and "
      "0 <= i0 <= 63 and 0 <= i1 <= 16 and 0 <= i2 <= 2 and 0 <= ct <= 1023 "
      "and 0 <= slot <= 4095 and 2048*floor((-30 - 30i0 + slot)/2048) >= -3967 "
      "+ slot and 2048*floor((-30 - 30i0 + slot)/2048) >= -2079 - 30i0 + i2 + "
      "slot and 2048*floor((-30 - 30i0 + slot)/2048) <= -2048 - 30i0 + slot "
      "and 2048*floor((-30 - 30i0 + slot)/2048) <= -2048 - 30i0 + i2 + slot "
      "and 2048*floor((-30 - 30i0 + slot)/2048) <= -2048 + slot }");
  ASSERT_TRUE(succeeded(relation));

  OpBuilder builder(&context);
  auto moduleOp = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = func::FuncOp::create(builder, builder.getUnknownLoc(),
                                     "test_func", funcType);
  auto* block = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(block);

  ImplicitLocOpBuilder locBuilder(builder.getUnknownLoc(), builder);
  auto init = arith::ConstantIntOp::create(locBuilder, 0, 32);

  MLIRLoopNestGenerator generator(locBuilder);
  auto bodyBuilder = [](OpBuilder& b, Location loc, ValueRange ivs,
                        ValueRange iterArgs) {
    return scf::ValueVector(iterArgs.begin(), iterArgs.end());
  };

  SmallVector<int> domainIndicesToSchedule = {0, 1};
  auto result = generator.generateForLoop(relation.value(), {init.getResult()},
                                          bodyBuilder, domainIndicesToSchedule);
  ASSERT_TRUE(succeeded(result));

  func::ReturnOp::create(locBuilder, ValueRange{});

  ASSERT_TRUE(succeeded(verify(moduleOp)));

  // Each guard is an i1 taken straight from the comparison logic.
  int numIfs = 0;
  funcOp.walk([&](scf::IfOp ifOp) {
    ++numIfs;
    Value cond = ifOp.getCondition();
    EXPECT_TRUE(cond.getType().isInteger(1));
    Operation* condOp = cond.getDefiningOp();
    EXPECT_TRUE(condOp != nullptr &&
                (isa<arith::CmpIOp, arith::AndIOp, arith::OrIOp>(condOp)))
        << "scf.if guard should come from comparison logic, got "
        << (condOp ? condOp->getName().getStringRef().str() : "block argument");
  });
  // Guard against the checks above going vacuous if ISL stops emitting `if`s.
  EXPECT_GT(numIfs, 0) << "relation generated no scf.if guards";

  // No boolean is round-tripped through index.
  funcOp.walk([&](arith::IndexCastOp castOp) {
    EXPECT_FALSE(castOp->getOperand(0).getType().isInteger(1))
        << "index_cast widens an i1 boolean to index";
    EXPECT_FALSE(castOp->getResult(0).getType().isInteger(1))
        << "index_cast narrows an index back to an i1 boolean";
  });

  moduleOp.erase();
}

TEST(CodegenTest, Conv2dChwFchwAsSequenceTest) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 28, 28}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  int64_t minSlotCount = 4096;

  auto maybeRels = get2dConvChwFchwFilterAsSequence(
      filterType, dataType, strides, padding, minSlotCount, true);
  ASSERT_TRUE(succeeded(maybeRels));
  auto rels = maybeRels.value();

  // Scheduling f, c, h, w as outer loops to improve performance.
  for (const auto& rel : rels) {
    auto result = generateLoopNestAsCStr(rel, {});
    ASSERT_TRUE(succeeded(result));
  }
}

TEST(CodegenTest, Conv2dChwFchwAsSequenceNoInterchangeTest) {
  MLIRContext context;
  RankedTensorType filterType =
      RankedTensorType::get({4, 4, 2, 2}, IndexType::get(&context));
  RankedTensorType dataType =
      RankedTensorType::get({1, 4, 28, 28}, IndexType::get(&context));
  SmallVector<int64_t> strides = {2, 2};
  int64_t padding = 0;
  int64_t minSlotCount = 4096;

  auto maybeRels = get2dConvChwFchwFilterAsSequence(
      filterType, dataType, strides, padding, minSlotCount, false);
  ASSERT_TRUE(succeeded(maybeRels));
  auto rels = maybeRels.value();

  EXPECT_EQ(rels.size(),
            4);  // 4 steps when interchange is false (skips step 2)

  for (size_t i = 0; i < rels.size(); ++i) {
    auto result = generateLoopNestAsCStr(rels[i]);
    EXPECT_TRUE(succeeded(result)) << "Failed at relation " << i;
  }
}

// The points of `relation` whose coordinates lie in [lb, ub], found by testing
// every candidate for membership.
std::vector<std::vector<int64_t>> bruteForcePoints(
    const presburger::IntegerRelation& relation, ArrayRef<int64_t> lb,
    ArrayRef<int64_t> ub) {
  std::vector<std::vector<int64_t>> points;
  std::vector<int64_t> point(lb.begin(), lb.end());
  while (true) {
    if (relation.containsPointNoLocal(point)) points.push_back(point);
    int d = point.size() - 1;
    for (; d >= 0; --d) {
      if (++point[d] <= ub[d]) break;
      point[d] = lb[d];
    }
    if (d < 0) return points;
  }
}

std::vector<std::vector<int64_t>> scannedPoints(
    const presburger::IntegerRelation& relation) {
  std::vector<std::vector<int64_t>> points;
  forEachPoint(relation, [&](ArrayRef<int64_t> domainPoint,
                             ArrayRef<int64_t> rangePoint) {
    std::vector<int64_t> point(domainPoint.begin(), domainPoint.end());
    point.insert(point.end(), rangePoint.begin(), rangePoint.end());
    points.push_back(point);
  });
  return points;
}

TEST(CodegenTest, ForEachPointHaleviShoup) {
  auto relation = getIntegerRelationFromIslStr(
      "{ [row, col] -> [ct, slot] : (slot - row) mod 4 = 0 and (ct + slot - "
      "col) mod 8 = 0 and 0 <= row <= 3 and 0 <= col <= 7 and 0 <= ct <= 3 "
      "and 0 <= slot <= 15 }");
  ASSERT_TRUE(succeeded(relation));
  // The brute force lists the points in lexicographic order, as forEachPoint
  // must.
  EXPECT_THAT(
      scannedPoints(relation.value()),
      Eq(bruteForcePoints(relation.value(), {0, 0, 0, 0}, {3, 7, 3, 15})));
}

TEST(CodegenTest, ForEachPointFloorDivisions) {
  auto relation = getIntegerRelationFromIslStr(
      "{ [i] -> [ct, slot] : ct = floor(i/4) and slot = i mod 4 and 0 <= i <= "
      "9 }");
  ASSERT_TRUE(succeeded(relation));
  EXPECT_THAT(scannedPoints(relation.value()),
              Eq(bruteForcePoints(relation.value(), {0, 0, 0}, {9, 2, 3})));
}

TEST(CodegenTest, ForEachPointExistential) {
  // isl keeps e as a loop of the scan, and e takes two values for some points;
  // each point must still be reported once.
  auto relation = getIntegerRelationFromIslStr(
      "{ [i] -> [o] : exists (e : i <= 2e <= i + 3 and o <= 3e <= o + 4) and "
      "0 <= i <= 3 and 0 <= o <= 3 }");
  ASSERT_TRUE(succeeded(relation));
  EXPECT_THAT(scannedPoints(relation.value()),
              Eq(bruteForcePoints(relation.value(), {0, 0}, {3, 3})));
}

TEST(CodegenTest, ForEachPointLargeCoordinates) {
  // The loop counter passes INT64_MAX after the last point.
  auto relation = getIntegerRelationFromIslStr(
      "{ [i] -> [x] : i = 0 and 9223372036854775806 <= x <= "
      "9223372036854775807 }");
  ASSERT_TRUE(succeeded(relation));
  std::vector<std::vector<int64_t>> expected = {{0, 9223372036854775806},
                                                {0, 9223372036854775807}};
  EXPECT_THAT(scannedPoints(relation.value()), Eq(expected));
}

TEST(CodegenTest, ForEachPointLargeIntermediates) {
  // 4000000000 * i overflows int64_t before the subtraction brings it back.
  auto relation = getIntegerRelationFromIslStr(
      "{ [i] -> [o] : 4000000000 <= i <= 4000000002 and o = 4000000000i - "
      "8000000000000000000 }");
  ASSERT_TRUE(succeeded(relation));
  std::vector<std::vector<int64_t>> expected = {
      {4000000000, 8000000000000000000},
      {4000000001, 8000000004000000000},
      {4000000002, 8000000008000000000}};
  EXPECT_THAT(scannedPoints(relation.value()), Eq(expected));
}

TEST(CodegenTest, ForEachPointLargeConstants) {
  // ct = 4000000000i - 8000000000 and slot = 2000000000ct -
  // 8000000000000000000, built directly so that isl's parser keeps both
  // equalities. isl substitutes ct into slot, which gives the constant
  // -24000000000000000000.
  presburger::IntegerRelation relation(
      presburger::PresburgerSpace::getRelationSpace(/*numDomain=*/1,
                                                    /*numRange=*/2));
  relation.addBound(presburger::BoundType::LB, 0, 3);
  relation.addBound(presburger::BoundType::UB, 0, 4);
  relation.addEquality({-4000000000LL, 1, 0, 8000000000LL});
  relation.addEquality({0, -2000000000LL, 1, 8000000000000000000LL});
  std::vector<std::vector<int64_t>> expected = {
      {3, 4000000000, 0}, {4, 8000000000, 8000000000000000000}};
  EXPECT_THAT(scannedPoints(relation), Eq(expected));
}

TEST(CodegenTest, ForEachPointEmpty) {
  auto relation = getIntegerRelationFromIslStr(
      "{ [i] -> [o] : o = 2i and 0 <= i <= 4 and o >= 9 }");
  ASSERT_TRUE(succeeded(relation));
  EXPECT_TRUE(scannedPoints(relation.value()).empty());
}

}  // namespace
}  // namespace heir
}  // namespace mlir
