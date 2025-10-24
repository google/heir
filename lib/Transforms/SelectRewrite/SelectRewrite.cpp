#include "lib/Transforms/SelectRewrite/SelectRewrite.h"

#include <cassert>
#include <utility>

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SELECTREWRITE
#include "lib/Transforms/SelectRewrite/SelectRewrite.h.inc"

Value matchShapedType(OpBuilder builder, Value op, Value target) {
  auto targetShapedType = mlir::dyn_cast<ShapedType>(target.getType());
  auto opShapedType = mlir::dyn_cast<ShapedType>(op.getType());
  if (targetShapedType && !opShapedType) {
    auto newShapedType =
        targetShapedType.cloneWith(targetShapedType.getShape(), op.getType());
    return tensor::SplatOp::create(builder, op.getLoc(), op, newShapedType);
  }
  return op;
}

TypedAttr getMatchingOne(Value op) {
  auto intType = getElementTypeOrSelf(op);
  assert(intType.getIntOrFloatBitWidth() == 1 && "Expected i1 type");
  if (auto st = mlir::dyn_cast<ShapedType>(op.getType())) {
    return DenseElementsAttr::get(st, true);
  }
  return IntegerAttr::get(intType, 1);
}

namespace select_rewrite {
// In an inner namespace to avoid conflicts
#include "lib/Transforms/SelectRewrite/SelectRewrite.cpp.inc"
}  // namespace select_rewrite

struct SelectRewrite : impl::SelectRewriteBase<SelectRewrite> {
  using SelectRewriteBase::SelectRewriteBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    select_rewrite::populateWithGenerated(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
