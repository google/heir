#include <iostream>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Parser.h"
#include "mlir/include/mlir/IR/Builders.h"     // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::Identifier;
using presburger::IntegerRelation;
using presburger::VarKind;
using ::testing::ElementsAre;
using ::testing::Eq;

TEST(CodegenTest, PureAffineEquality) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  IntegerRelation relation = relationFromString(
      "(d0, d1) : (d0 - d1 == 0, d0 >= 0, d1 >= 0, 10 >= d0, 10 >= d1)", 1,
      &context);
  auto loopNestRes = generateLoopNest(relation, &context);
  ASSERT_TRUE(succeeded(loopNestRes));

  auto actual = loopNestRes.value();

  LoopNest expected;
  expected.numInductionVars = 2;
  expected.lowerBounds = {0, 0};
  expected.upperBounds = {10, 10};

  OpBuilder b(&context);
  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);
  expected.constraints.push_back(d0 - d1);
  expected.constraints.push_back(d0);
  expected.constraints.push_back(d1);

  ASSERT_THAT(actual, Eq(expected));
}

TEST(CodegenTest, EqualityWithMod) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  IntegerRelation relation = relationFromString(
      "(d0, d1) : ((d0 - d1) mod 2 == 0, d0 >= 0, d1 >= 0, 10 >= d0, 30 >= d1)",
      1, &context);
  std::string dataId = "data";
  std::string ctId = "ct";
  relation.resetIds();
  relation.setId(VarKind::Domain, 0, Identifier(&dataId));
  relation.setId(VarKind::Range, 0, Identifier(&ctId));

  auto loopNestRes = generateLoopNest(relation, &context);
  ASSERT_TRUE(succeeded(loopNestRes));

  auto actual = loopNestRes.value();

  LoopNest expected;
  expected.numInductionVars = 2;
  expected.lowerBounds = {0, 0};
  expected.upperBounds = {10, 30};

  // OpBuilder b(&context);
  // auto d0 = b.getAffineDimExpr(0);
  // auto d1 = b.getAffineDimExpr(1);
  // expected.constraints.push_back(d0 - d1);
  // expected.constraints.push_back(d1);
  // expected.constraints.push_back(10 - d1);

  ASSERT_THAT(actual.numInductionVars, Eq(2));
  EXPECT_THAT(actual.lowerBounds, ElementsAre(0, 0));
  EXPECT_THAT(actual.upperBounds, ElementsAre(10, 30));
}
}  // namespace
}  // namespace heir
}  // namespace mlir
