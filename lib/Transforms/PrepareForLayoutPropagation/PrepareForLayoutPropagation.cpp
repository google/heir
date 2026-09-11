#include "lib/Transforms/PrepareForLayoutPropagation/PrepareForLayoutPropagation.h"

#include <memory>

#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"         // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PREPAREFORLAYOUTPROPAGATION
#include "lib/Transforms/PrepareForLayoutPropagation/PrepareForLayoutPropagation.h.inc"

namespace {

// Raise a broadcast-shaped linalg.generic to the named linalg.broadcast.
struct GenericToBroadcast : public OpRewritePattern<linalg::GenericOp> {
 public:
  GenericToBroadcast(MLIRContext* context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
        op.getNumResults() != 1)
      return failure();
    if (llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
          return t != utils::IteratorType::parallel;
        }))
      return failure();

    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity()) return failure();
    AffineMap inputMap = maps[0];

    // The input map must be a strictly increasing projection of the output
    // dims, e.g. (d0, d1) -> (d0). The broadcast dimensions are its
    // complement.
    SmallVector<int64_t> inputDims;
    int64_t prev = -1;
    for (AffineExpr expr : inputMap.getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim || int64_t(dim.getPosition()) <= prev) return failure();
      prev = dim.getPosition();
      inputDims.push_back(prev);
    }
    SmallVector<int64_t> broadcastDims;
    for (int64_t d = 0; d < int64_t(inputMap.getNumDims()); ++d) {
      if (!llvm::is_contained(inputDims, d)) broadcastDims.push_back(d);
    }
    if (broadcastDims.empty()) return failure();

    // Body must be a pure pass-through of the input block argument.
    Block& block = op.getRegion().front();
    if (!block.without_terminator().empty()) return failure();
    auto yield = cast<linalg::YieldOp>(block.getTerminator());
    if (yield.getNumOperands() != 1 ||
        yield.getOperand(0) != block.getArgument(0))
      return failure();

    auto bcast =
        linalg::BroadcastOp::create(rewriter, op.getLoc(), op.getInputs()[0],
                                    op.getDpsInits()[0], broadcastDims);
    rewriter.replaceOp(op, bcast.getResults());
    return success();
  }
};

// Rewrite x / splat-constant into x * splat(1/constant).
struct DivfByConstantToMulf : public OpRewritePattern<arith::DivFOp> {
 public:
  DivfByConstantToMulf(MLIRContext* context)
      : OpRewritePattern<arith::DivFOp>(context) {}

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter& rewriter) const override {
    DenseElementsAttr attr;
    if (!matchPattern(op.getRhs(), m_Constant(&attr)))
      return rewriter.notifyMatchFailure(op, "rhs is not a constant");
    auto splat = dyn_cast<SplatElementsAttr>(attr);
    if (!splat) return rewriter.notifyMatchFailure(op, "rhs is not a splat");
    auto floatAttr = dyn_cast<FloatAttr>(splat.getSplatValue<Attribute>());
    if (!floatAttr) return failure();
    double value = floatAttr.getValueAsDouble();
    if (value == 0.0) return failure();

    auto type = cast<ShapedType>(op.getRhs().getType());
    auto recipAttr = DenseElementsAttr::get(
        type, rewriter.getFloatAttr(type.getElementType(), 1.0 / value));
    Value recip =
        arith::ConstantOp::create(rewriter, op.getLoc(), type, recipAttr);
    Value mul =
        arith::MulFOp::create(rewriter, op.getLoc(), op.getLhs(), recip);
    rewriter.replaceOp(op, mul);
    return success();
  }
};

}  // namespace

struct PrepareForLayoutPropagation
    : public impl::PrepareForLayoutPropagationBase<
          PrepareForLayoutPropagation> {
  using PrepareForLayoutPropagationBase::PrepareForLayoutPropagationBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<GenericToBroadcast, DivfByConstantToMulf>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
