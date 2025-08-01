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

using ::testing::Eq;

// FIXME: this test should require only one loop because d0 == d1
TEST(CodegenTest, NaiveExpensiveLoop) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  presburger::IntegerRelation relation = relationFromString(
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
  expected.constraints.push_back(10 - d0);
  expected.constraints.push_back(10 - d1);

  ASSERT_THAT(actual, Eq(expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
