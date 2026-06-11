#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project

namespace mlir {
namespace heir {

namespace {

TEST(UtilsTest, ExtendToCommonWidthI1ToI32) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  Block block;
  builder.setInsertionPointToStart(&block);
  Value c1 = arith::ConstantIntOp::create(builder, loc, 1, 1).getResult();
  Value c32 = arith::ConstantIntOp::create(builder, loc, 42, 32).getResult();
  SmallVector<Value> values = {c1, c32};

  SmallVector<Value> extended = extendToCommonWidth(builder, values, nullptr);
  ASSERT_EQ(extended.size(), 2);

  // i1 should be extended to i32, i32 should be unchanged.
  EXPECT_EQ(extended[0].getType(), builder.getI32Type());
  EXPECT_EQ(extended[1], c32);

  // First value is extended via arith::ExtUIOp (unsigned)
  auto extOp = extended[0].getDefiningOp();
  ASSERT_TRUE(extOp != nullptr);
  EXPECT_TRUE(isa<arith::ExtUIOp>(extOp));
}

}  // namespace

}  // namespace heir
}  // namespace mlir
