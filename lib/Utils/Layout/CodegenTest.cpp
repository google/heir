#include <cstdlib>
#include <string>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Parser.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

// ISL
#include "include/isl/ast.h"       // from @isl
#include "include/isl/ast_type.h"  // from @isl
#include "include/isl/ctx.h"       // from @isl

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;
using ::testing::Eq;

FailureOr<std::string> runAndGetCStr(const IntegerRelation& relation) {
  isl_ctx* ctx = isl_ctx_alloc();
  auto result = generateLoopNest(relation, ctx);
  if (failed(result)) {
    isl_ctx_free(ctx);
    return failure();
  }
  isl_ast_node* tree = result.value();
  char* c_str = isl_ast_node_to_C_str(tree);
  std::string actual = std::string(c_str);
  free(c_str);
  isl_ast_node_free(tree);
  isl_ctx_free(ctx);
  // Add a leading newline for ease of comparison with multiline strings.
  return actual.insert(0, "\n");
}

TEST(CodegenTest, PureAffineEquality) {
  MLIRContext context;
  IntegerRelation relation = relationFromString(
      "(d0, d1) : (d0 - d1 == 0, d0 >= 0, d1 >= 0, 10 >= d0, 10 >= d1)", 1,
      &context);
  auto result = runAndGetCStr(relation);
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
  IntegerRelation relation = relationFromString(
      "(d0, d1) : ((d0 - d1) mod 2 == 0, d0 >= 0, d1 >= 0, 10 >= d0, 30 >= d1)",
      1, &context);

  auto result = runAndGetCStr(relation);
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
  IntegerRelation relation = relationFromString(
      "(row, col, ct, slot) : ((slot - row) mod 32 == 0, (ct + slot - col) mod "
      "64 == 0, row >= 0, col >= 0, ct >= 0, slot >= 0, 1023 >= slot, 31 >= "
      "ct, 31 >= row, 63 >= col)",
      2, &context);
  auto result = runAndGetCStr(relation);
  ASSERT_TRUE(succeeded(result));
  std::string actual = result.value();
  std::string expected = R"(
for (int c0 = 0; c0 <= 31; c0 += 1)
  for (int c1 = 0; c1 <= 1023; c1 += 1)
    S(c1 % 32, -((-c0 - c1 + 1087) % 64) + 63, c0, c1);
)";
  ASSERT_THAT(actual, Eq(expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
