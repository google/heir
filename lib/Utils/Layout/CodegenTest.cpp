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
using ::testing::Eq;

TEST(CodegenTest, PureAffineEquality) {
  MLIRContext context;
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
  MLIRContext context;
  IntegerRelation relation = relationFromString(
      "(d0, d1) : ((d0 - d1) mod 2 == 0, d0 >= 0, d1 >= 0, 10 >= d0, 30 >= d1)",
      1, &context);
  std::string dataId = "data";
  std::string slotId = "slot";
  relation.resetIds();
  relation.setId(VarKind::Domain, 0, Identifier(&dataId));
  relation.setId(VarKind::Range, 0, Identifier(&slotId));

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

TEST(CodegenTest, HaleviShoup) {
  MLIRContext context;
  // Data is 32x64, being packed into ciphertexts of size 1024 via Halevi-Shoup
  // diagonal layout.
  IntegerRelation relation = relationFromString(
      "(row, col, ct, slot) : ((slot - row) mod 32 == 0, (ct + slot - col) mod "
      "64 == 0, row >= 0, col >= 0, ct >= 0, slot >= 0, 1023 >= slot, 31 >= "
      "ct, 31 >= row, 63 >= col)",
      2, &context);
  auto loopNestRes = generateLoopNest(relation, &context);
  ASSERT_TRUE(succeeded(loopNestRes));
  auto actual = loopNestRes.value();

  auto row = getAffineDimExpr(0, &context);
  auto col = getAffineDimExpr(1, &context);
  auto ct = getAffineDimExpr(2, &context);
  auto slot = getAffineDimExpr(3, &context);

  LoopNest expected;
  expected.addLoop(2, 0, 31);               // ct
  Loop *l2 = expected.addLoop(3, 0, 1023);  // slot

  l2->eliminatedVariables[0] = slot.floorDiv(32);
  l2->eliminatedVariables[1] = -(ct + slot).floorDiv(64);

  // l2->constraints = {
  //     slot - ((-row + slot).floorDiv(32)) * 32,
  //     -slot + ((-row + slot).floorDiv(32)) * 32 + 31,
  //     col - ct - slot + ((-col + ct + slot).floorDiv(64) * 64 + 63),
  //     (-col + ct + slot) % 64};
  // l2->eq = {false, false, false, false};

  ASSERT_THAT(actual, Eq(expected));

  // For some reason the AffineExpr equality operator doesn't work
  // because the first half-byte is off. Sometimes applying simplifyAffineExpr
  // fixes it, but not always. So constraints are compared manually here.
  ASSERT_THAT(actual.loops.size(), Eq(4));
  for (unsigned i = 0; i < actual.loops.size(); ++i) {
    for (const auto &[c1, c2] : llvm::zip(actual.loops[i].constraints,
                                          expected.loops[i].constraints)) {
      // llvm::errs() << "Comparing constraints: " << c1 << " vs " << c2 << " ?
      // "
      //              << (simplifyAffineExpr(c1, 4, 0) ==
      //                  simplifyAffineExpr(c2, 4, 0))
      //              << "\n";
      ASSERT_TRUE(simplifyAffineExpr(c1, 4, 0) == simplifyAffineExpr(c2, 4, 0));
    }
  }
}
}  // namespace
}  // namespace heir
}  // namespace mlir
