#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"

#include <cstddef>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ELEMENTWISETOAFFINE
#include "lib/Transforms/ElementwiseToAffine/ElementwiseToAffine.h.inc"

// All of this is based on the ElementwiseToLinalg Pass in
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp

static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op)) return false;

  return llvm::any_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

namespace {

struct ConvertAnyElementwiseMappableOpOnRankedTensors : public RewritePattern {
  ConvertAnyElementwiseMappableOpOnRankedTensors(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    if (!isElementwiseMappableOpOnRankedTensors(op))
      return rewriter.notifyMatchFailure(
          op, "requires elementwise op on ranked tensors");

    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    auto elementType = resultType.getElementType();
    auto shape = resultType.getShape();
    auto rank = resultType.getRank();

    // Save insertion point prior to entering loop nest
    auto ip = rewriter.saveInsertionPoint();

    // Create an empty tensor as initial value of the iter_args
    Value target =
        rewriter.create<tensor::EmptyOp>(op->getLoc(), shape, elementType);

    llvm::SmallVector<Value, 1> indices;

    // Create a an affine.for loop nest of depth rank
    for (size_t i = 0; i < rank; ++i) {
      auto loop =
          rewriter.create<affine::AffineForOp>(op->getLoc(), /* lowerBound*/ 0,
                                               /* upperBound*/ shape[i],
                                               /* step*/ 1,
                                               /* iterArgs*/ target);

      // Update target & indices
      target = loop.getRegionIterArgs().front();
      indices.push_back(loop.getInductionVar());

      // If first loop: replace scalar op
      if (i == 0) {
        rewriter.replaceOp(op, loop);
      } else {  // yield the result of this loop
        rewriter.create<affine::AffineYieldOp>(op->getLoc(),
                                               loop->getResults());
      }
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Create the innermost body
    auto resultTypes =
        llvm::to_vector<6>(llvm::map_range(op->getResultTypes(), [](Type type) {
          return cast<TensorType>(type).getElementType();
        }));

    // Generate a `tensor.extract` for each tensor operand
    SmallVector<Value, 4> newOperands;
    for (auto operand : op->getOperands()) {
      if (mlir::isa<RankedTensorType>(operand.getType())) {
        // We don't need to check the shape, as ElementwiseMappable
        // requires all tensor operands to have compatible shapes
        auto extractOp = rewriter.create<tensor::ExtractOp>(operand.getLoc(),
                                                            operand, indices);
        newOperands.push_back(extractOp);
      } else {
        // scalar (technically, "non-tensor") operands can be reused as-is
        newOperands.push_back(operand);
      }
    }

    // "lowered" operation is the same operation, but over non-tensor
    // operands
    auto *scalarOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        newOperands, resultTypes, op->getAttrs());

    // insert scalarOp into the tensor at right index
    Value inserted = rewriter.create<tensor::InsertOp>(
        op->getLoc(), scalarOp->getResult(0), target, indices);

    // replace lingalg.yield scalarOp with affine.yield insertedOp
    rewriter.create<affine::AffineYieldOp>(op->getLoc(), inserted);

    // reset insertion point
    rewriter.restoreInsertionPoint(ip);

    return success();
  }
};
}  // namespace

struct ElementwiseToAffine
    : impl::ElementwiseToAffineBase<ElementwiseToAffine> {
  using ElementwiseToAffineBase::ElementwiseToAffineBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    patterns.add<ConvertAnyElementwiseMappableOpOnRankedTensors>(context);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      bool convertAll = convertDialects.empty() && convertOps.empty();
      bool convertDialect = llvm::is_contained(
          convertDialects, op->getDialect()->getNamespace().str());
      bool convertOp =
          llvm::is_contained(convertOps, op->getName().getStringRef().str());

      if (convertAll || convertDialect || convertOp)
        return !isElementwiseMappableOpOnRankedTensors(op);
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
