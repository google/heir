#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/IRMaterializingVisitor.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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
    rewriter.replaceOp(op, finalOutput);
    return success();
  }
};

WalkResult handleInferTypeOpInterface(InferTypeOpInterface op) {
  SmallVector<Type, 4> expectedResultTypes;
  LogicalResult result = op.inferReturnTypes(
      op->getContext(), op.getLoc(), op->getOperands(), op->getAttrDictionary(),
      op->getPropertiesStorage(), op->getRegions(), expectedResultTypes);

  if (failed(result)) {
    op->emitError() << "Failed to infer types for " << op->getName()
                    << " during Orion to CKKS conversion.";
    return WalkResult::interrupt();
  }

  for (auto [result, expectedType] :
       llvm::zip(op->getResults(), expectedResultTypes)) {
    if (result.getType() != expectedType) {
      result.setType(expectedType);
    }
  }
  return WalkResult::advance();
}

WalkResult handleRescaleOp(ckks::RescaleOp op) {
  lwe::LWECiphertextType inputType =
      cast<lwe::LWECiphertextType>(op.getInput().getType());
  FailureOr<lwe::LWECiphertextType> outputTypeResult =
      applyModReduce(inputType);
  if (failed(outputTypeResult)) {
    op.emitError() << "Cannot drop one limb from ciphertext type: "
                   << inputType;
    return WalkResult::interrupt();
  }
  op.getResult().setType(outputTypeResult.value());
  op.setToRingAttr(outputTypeResult.value().getCiphertextSpace().getRing());
  return WalkResult::advance();
}

template <typename CtPtOp>
WalkResult handleNonMulCtPtOp(CtPtOp op) {
  lwe::LWECiphertextType ctType;
  if (isa<lwe::LWECiphertextType>(op.getLhs().getType())) {
    ctType = cast<lwe::LWECiphertextType>(op.getLhs().getType());
  } else {
    ctType = cast<lwe::LWECiphertextType>(op.getRhs().getType());
  }
  op.getResult().setType(ctType);
  return WalkResult::advance();
}

WalkResult handleMulPlain(ckks::MulPlainOp op) {
  lwe::LWECiphertextType ctType;
  lwe::LWEPlaintextType ptType;
  if (isa<lwe::LWECiphertextType>(op.getLhs().getType())) {
    ctType = cast<lwe::LWECiphertextType>(op.getLhs().getType());
    ptType = cast<lwe::LWEPlaintextType>(op.getRhs().getType());
  } else {
    ctType = cast<lwe::LWECiphertextType>(op.getRhs().getType());
    ptType = cast<lwe::LWEPlaintextType>(op.getLhs().getType());
  }
  auto newCtType = lwe::LWECiphertextType::get(
      op.getContext(), ctType.getApplicationData(),
      inferMulOpPlaintextSpaceAttr(op.getContext(), ctType.getPlaintextSpace(),
                                   ptType.getPlaintextSpace()),
      ctType.getCiphertextSpace(), ctType.getKey(), ctType.getModulusChain());
  op.getResult().setType(newCtType);
  return WalkResult::advance();
}

struct OrionToCKKS : public impl::OrionToCKKSBase<OrionToCKKS> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertChebyshevOp, ConvertLinearTransformOp>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
    Operation* root = getOperation();

    // At this step, the types are wrong and need to be re-propagated In
    // particular, mul and mul_plain ops are followed by a rescale, and while
    // the result type drops a limb, the downstream ops are not updated to
    // match.
    WalkResult walkResult = root->walk([&](Operation* op) {
      if (op->hasTrait<OpTrait::SameOperandsAndResultType>()) {
        if (!llvm::all_equal(op->getOperandTypes())) {
          root->emitError()
              << "Operand types do not match for " << op->getName();
        }
        for (OpResult result : op->getResults()) {
          result.setType(op->getOperand(0).getType());
        }
        return WalkResult::advance();
      }

      return llvm::TypeSwitch<Operation*, WalkResult>(op)
          .Case<InferTypeOpInterface>(handleInferTypeOpInterface)
          .Case<ckks::RescaleOp>(handleRescaleOp)
          .Case<ckks::AddPlainOp>(handleNonMulCtPtOp<ckks::AddPlainOp>)
          .Case<ckks::SubPlainOp>(handleNonMulCtPtOp<ckks::SubPlainOp>)
          .Case<ckks::MulPlainOp>(handleMulPlain)
          .Default([&](Operation* op) {
            if (llvm::none_of(op->getResults(), [](auto result) {
                  return isa<lwe::LWECiphertextType>(result.getType());
                })) {
              // No ciphertext types, so no type propagation needed
              return WalkResult::advance();
            }
            op->emitError() << "Unhandled operation type " << op->getName()
                            << " during Orion to CKKS type re-propagation.";
            return WalkResult::interrupt();
          });
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }

    // Now propagate the final return types to the function's signature.
    root->walk([&](func::FuncOp funcOp) {
      func::ReturnOp returnOp =
          cast<func::ReturnOp>(funcOp.getBody().back().getTerminator());
      FunctionType newFuncType = FunctionType::get(
          funcOp.getContext(), funcOp.getFunctionType().getInputs(),
          returnOp.getOperandTypes());
      funcOp.setType(newFuncType);
    });
  }
};

}  // namespace mlir::heir::orion
