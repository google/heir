#include <cstdlib>
#include <string>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Parser.h"
#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

// ISL
#include "include/isl/ctx.h"       // from @isl
#include "include/isl/map.h"       // from @isl
#include "include/isl/map_type.h"  // from @isl

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;
using presburger::VarKind;

TEST(IslConversionTest, SimpleTest) {
  MLIRContext context;
  std::string relStr =
      "(d0, d1) : ((d0 - d1) mod 7 == 0, d0 >= 0, 20 >= d0, d1 >= 0, 20 >= d1)";
  IntegerRelation relation =
      affine::FlatAffineValueConstraints(parseIntegerSet(relStr, &context));
  relation.convertVarKind(VarKind::SetDim, 0, 1, VarKind::Domain);

  std::string expected =
      "{ [i0] -> [o0] : (-i0 + o0) mod 7 = 0 and 0 <= i0 <= 20 and 0 <= o0 <= "
      "20 }";

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* result = convertRelationToBasicMap(relation, ctx);

  char* resultStr = isl_basic_map_to_str(result);
  std::string actual(resultStr);
  free(resultStr);
  EXPECT_EQ(expected, actual);

  isl_basic_map_free(result);
  isl_ctx_free(ctx);
}

TEST(IslConversionTest, RegressionTest) {
  MLIRContext context;
  std::string relStr =
      "(d0, d1, d2, d3, d4, d5, d6, d7) : (-d1 + d3 == 0, d0 - d4 * 1024 - d5 "
      "== 0, d2 - d6 * 32 - d7 == 0, d5 - d7 == 0, -d0 + 31 >= 0, d0 >= 0, d1 "
      "== 0, d2 >= 0, -d2 + 1023 >= 0, -d0 + d3 * 1024 + 1023 >= 0, d0 - d3 * "
      "1024 >= 0, -d0 + d4 * 1024 + 1023 >= 0, d0 - d4 * 1024 >= 0, -d2 + d6 * "
      "32 + 31 >= 0, d2 - d6 * 32 >= 0)";
  IntegerRelation relation =
      affine::FlatAffineValueConstraints(parseIntegerSet(relStr, &context));
  relation.convertVarKind(VarKind::SetDim, 3, 8, VarKind::Local);
  relation.convertVarKind(VarKind::SetDim, 0, 1, VarKind::Domain);

  std::string expected =
      "{ [i0] -> [o0, o1] : o0 = 0 and (-i0 + o1) mod 32 = 0 and 0 <= i0 <= 31 "
      "and 0 <= o1 <= 1023 }";

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* result = convertRelationToBasicMap(relation, ctx);

  char* resultStr = isl_basic_map_to_str(result);
  std::string actual(resultStr);
  free(resultStr);
  EXPECT_EQ(expected, actual);

  isl_basic_map_free(result);
  isl_ctx_free(ctx);
}

TEST(IslConversionTest, SimpleRoundTrip) {
  MLIRContext context;
  std::string relStr =
      "(d0, d1) : ((d0 - d1) mod 7 == 0, d0 >= 0, 20 >= d0, d1 >= 0, 20 >= d1)";
  IntegerRelation relation =
      affine::FlatAffineValueConstraints(parseIntegerSet(relStr, &context));
  relation.convertVarKind(VarKind::SetDim, 0, 1, VarKind::Domain);

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);
  IntegerRelation actual = convertBasicMapToRelation(bmap);

  ASSERT_TRUE(relation.isEqual(actual));
}

TEST(IslConversionTest, RoundTripRegressionTest) {
  MLIRContext context;
  std::string relStr =
      "(d0, d1, d2, d3, d4, d5, d6, d7) : (-d1 + d3 == 0, d0 - d4 * 1024 - d5 "
      "== 0, d2 - d6 * 32 - d7 == 0, d5 - d7 == 0, -d0 + 31 >= 0, d0 >= 0, d1 "
      "== 0, d2 >= 0, -d2 + 1023 >= 0, -d0 + d3 * 1024 + 1023 >= 0, d0 - d3 * "
      "1024 >= 0, -d0 + d4 * 1024 + 1023 >= 0, d0 - d4 * 1024 >= 0, -d2 + d6 * "
      "32 + 31 >= 0, d2 - d6 * 32 >= 0)";
  IntegerRelation relation =
      affine::FlatAffineValueConstraints(parseIntegerSet(relStr, &context));
  relation.convertVarKind(VarKind::SetDim, 3, 8, VarKind::Local);
  relation.convertVarKind(VarKind::SetDim, 0, 1, VarKind::Domain);

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);
  IntegerRelation actual = convertBasicMapToRelation(bmap);

  ASSERT_TRUE(relation.isEqual(actual));
}

TEST(IslConversionTest, ScalarLayout) {
  MLIRContext context;
  std::string relStr =
      "(ct, slot) : (0 <= ct, 0 >= ct, slot >=0, 1023 >= slot)";
  IntegerRelation relation =
      affine::FlatAffineValueConstraints(parseIntegerSet(relStr, &context));
  // No domain vars, set dims are range vars by default

  std::string expected = "{ [] -> [o0, o1] : o0 = 0 and 0 <= o1 <= 1023 }";

  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* result = convertRelationToBasicMap(relation, ctx);

  char* resultStr = isl_basic_map_to_str(result);
  std::string actual(resultStr);
  free(resultStr);
  EXPECT_EQ(expected, actual);

  isl_basic_map_free(result);
  isl_ctx_free(ctx);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
