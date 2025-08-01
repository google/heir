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

TEST(CodegenTest, SimpleTest) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  presburger::IntegerRelation relation = relationFromString(
      "(d0, d1) : (d0 - d1 == 0, d0 >= 0, d1 >= 0, 10 >= d0, 10 >= d1)", 1,
      &context);
  auto loopNest = generateLoopNest(relation, &context);

  LoopNest expected;
  expected.numInductionVars = 2;
  expected.lowerBounds = {0, 0};
  expected.upperBounds = {10, 10};

  OpBuilder b(&context);
  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);
  expected.constraints.push_back(d1 - d0);

  ASSERT_THAT(loopNest, Eq(expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
