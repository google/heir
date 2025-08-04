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
  Loop *loop = expected.addLoop(1, 0, 10);                       // d1
  loop->eliminatedVariables[0] = getAffineDimExpr(1, &context);  // d0 = d1

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

  OpBuilder b(&context);
  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);

  LoopNest expected;
  expected.addLoop(1, 0, 30);                 // d1
  Loop *loopD1 = expected.addLoop(0, 0, 10);  // d0
  loopD1->eliminatedVariables[2] = (d0 - d1).floorDiv(2);

  loopD1->constraints.push_back((d0 - d1) % 2);
  loopD1->eq.push_back(true);

  // The two constraints below should be redundant given d0 - d1 mod 2 above,
  // but I'm not sure how to detect that systematically in the implementation.
  loopD1->constraints.push_back(d1 + ((d0 - d1).floorDiv(2)) * 2);
  loopD1->eq.push_back(false);

  loopD1->constraints.push_back(-d1 - ((d0 - d1).floorDiv(2)) * 2 + 10);
  loopD1->eq.push_back(false);

  ASSERT_THAT(actual, Eq(expected));
}
}  // namespace
}  // namespace heir
}  // namespace mlir
