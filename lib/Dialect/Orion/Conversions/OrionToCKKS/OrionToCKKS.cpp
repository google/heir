#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/IRMaterializingVisitor.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::orion {

using kernel::ArithmeticDagNode;
using kernel::implementHaleviShoup;
using kernel::SSAValue;

#define GEN_PASS_DEF_ORIONTOCKKS
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

struct ConvertChebyshevOp : public OpRewritePattern<ChebyshevOp> {
  using OpRewritePattern<ChebyshevOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ChebyshevOp op,
                                PatternRewriter& rewriter) const override {
    // FIXME: implement
    return failure();
  }
};

struct ConvertLinearTransformOp : public OpRewritePattern<LinearTransformOp> {
  using OpRewritePattern<LinearTransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearTransformOp op,
                                PatternRewriter& rewriter) const override {
    Value input = op.getInput();
    TypedValue<RankedTensorType> diagonals = op.getDiagonals();

    SSAValue vectorLeaf(input);
    SSAValue matrixLeaf(diagonals);
    std::shared_ptr<ArithmeticDagNode<SSAValue>> implementedKernel =
        implementHaleviShoup(vectorLeaf, matrixLeaf,
                             diagonals.getType().getShape());

    rewriter.setInsertionPointAfter(op);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    IRMaterializingVisitor visitor(b);
    Value finalOutput = implementedKernel->visit(visitor);

    // FIXME: propagate result type through the rest of the IR
    rewriter.replaceOp(op, finalOutput);
    return success();
  }
};

struct OrionToCKKS : public impl::OrionToCKKSBase<OrionToCKKS> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertChebyshevOp, ConvertLinearTransformOp>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace mlir::heir::orion
