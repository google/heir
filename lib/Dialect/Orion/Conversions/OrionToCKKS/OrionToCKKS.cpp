#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/IRMaterializingVisitor.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "orion-to-ckks"

namespace mlir::heir::orion {

using kernel::ArithmeticDagNode;
using kernel::implementHaleviShoup;
using kernel::SSAValue;

#define GEN_PASS_DEF_ORIONTOCKKS
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

constexpr int64_t defaultScalingFactor = 26;

lwe::LWEPlaintextType getPlaintextTypeFromCtTypeAndScalingFactor(
    lwe::LWECiphertextType ctTy, int64_t scalingFactor) {
  MLIRContext* ctx = ctTy.getContext();
  auto encodingAttr =
      lwe::InverseCanonicalEncodingAttr::get(ctx, defaultScalingFactor);
  return lwe::LWEPlaintextType::get(
      ctx, ctTy.getApplicationData(),
      lwe::PlaintextSpaceAttr::get(ctx, ctTy.getPlaintextSpace().getRing(),
                                   encodingAttr));
}

Value encodeSplattedCleartextUsingCtAndScalingFactor(
    ImplicitLocOpBuilder& b, lwe::LWECiphertextType ctTy, int64_t scalingFactor,
    APFloat value) {
  MLIRContext* ctx = b.getContext();
  lwe::LWEPlaintextType ptTy =
      getPlaintextTypeFromCtTypeAndScalingFactor(ctTy, scalingFactor);
  int64_t numSlots = ctTy.getCiphertextSpace()
                         .getRing()
                         .getPolynomialModulus()
                         .getPolynomial()
                         .getDegree() /
                     2;
  RankedTensorType cleartextType =
      RankedTensorType::get({numSlots}, Float32Type::get(ctx));

  // Create a mul_plain op with appropriate scaling factor on the encoded
  // plaintext constant
  auto constantOp = arith::ConstantOp::create(
      b, getScalarOrDenseAttr(cleartextType, std::move(value)));
  auto encodeOp = lwe::RLWEEncodeOp::create(
      b, ptTy, constantOp.getResult(), ptTy.getPlaintextSpace().getEncoding(),
      ptTy.getPlaintextSpace().getRing());
  return encodeOp.getResult();
}

struct ConvertChebyshevOp : public OpRewritePattern<ChebyshevOp> {
  using OpRewritePattern<ChebyshevOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ChebyshevOp op,
                                PatternRewriter& rewriter) const override {
    ArrayAttr chebCoeffsAttr = op.getCoefficients();
    SmallVector<double> chebCoeffs =
        llvm::map_to_vector(chebCoeffsAttr, [](Attribute attr) {
          return llvm::cast<FloatAttr>(attr).getValue().convertToDouble();
        });

    double lower = op.getDomainStartAttr().getValue().convertToDouble();
    double upper = op.getDomainEndAttr().getValue().convertToDouble();

    // The Chebyshev polynomial is defined on the interval [-1, 1]. We need to
    // rescale the input x in [lower, upper] to be on this unit interval.
    // The mapping is x -> 2(x-L)/(U-L) - 1 = (2/U-L) * x - (U+L)/(U-L)
    APFloat rescale = APFloat(2 / (upper - lower));
    APFloat shift = APFloat((upper + lower) / (upper - lower));

    lwe::LWECiphertextType ctTy = op.getInput().getType();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value xInput = op.getInput();

    if (!rescale.isExactlyValue(1.0)) {
      // The scaling factor here can be anything, so long as it has enough
      // precision to support the input. Default of 26 bits should be fine
      // for evaluation purposes.
      auto encodedSplatRescale = encodeSplattedCleartextUsingCtAndScalingFactor(
          b, ctTy, defaultScalingFactor, rescale);
      xInput =
          ckks::MulPlainOp::create(b, xInput, encodedSplatRescale).getResult();
    }

    if (!shift.isZero()) {
      // unlike mul_plain, add_plain requires the encoded plaintext to have
      // the same scaling factor as the ciphertext.
      int64_t scalingFactor = lwe::getScalingFactorFromEncodingAttr(
          ctTy.getPlaintextSpace().getEncoding());
      auto encodedSplatShift = encodeSplattedCleartextUsingCtAndScalingFactor(
          b, ctTy, scalingFactor, shift);
      xInput =
          ckks::AddPlainOp::create(b, xInput, encodedSplatShift).getResult();
    }

    SSAValue xNode(xInput);
    auto implementedKernel =
        polynomial::patersonStockmeyerChebyshevPolynomialEvaluation(xNode,
                                                                    chebCoeffs);
    IRMaterializingVisitor visitor(
        b,
        getPlaintextTypeFromCtTypeAndScalingFactor(ctTy, defaultScalingFactor));
    Value finalOutput = implementedKernel->visit(visitor);
    rewriter.replaceOp(op, finalOutput);
    return success();
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

WalkResult handleBootstrap(ckks::BootstrapOp op) {
  // First, we need to find the maximum level in the modulus chain from the
  // ciphertext type.
  lwe::LWECiphertextType inputType =
      cast<lwe::LWECiphertextType>(op.getInput().getType());
  // sub 1 because the max level is the last index in the chain.
  int64_t maxLevel = inputType.getModulusChain().getElements().size() - 1;

  // Now we cheat a little bit: normally bootstrap itself would consume some
  // levels, which depends on the chosen backend. In our case, we're lowering to
  // library backends that handle this opaquely, so as long as we keep the type
  // system consistent at this step, we're fine.
  //
  // TODO(#1207): fix if this pass still matters when lowering to polynomial.
  FailureOr<lwe::LWECiphertextType> outputTypeResult =
      cloneAtLevel(inputType, maxLevel);
  op.getResult().setType(outputTypeResult.value());
  return WalkResult::advance();
}

template <typename CtPtOp>
WalkResult handleNonMulCtPtOp(CtPtOp op) {
  lwe::LWECiphertextType ctType;
  Value plaintextOperand;
  if (isa<lwe::LWECiphertextType>(op.getLhs().getType())) {
    ctType = cast<lwe::LWECiphertextType>(op.getLhs().getType());
    plaintextOperand = op.getRhs();
  } else {
    ctType = cast<lwe::LWECiphertextType>(op.getRhs().getType());
    plaintextOperand = op.getLhs();
  }

  if (auto encodeOp =
          dyn_cast<lwe::RLWEEncodeOp>(plaintextOperand.getDefiningOp())) {
    // The plaintext operand is already encoded, so we need to set its
    // scaling factor to match the ciphertext operand.
    lwe::LWEPlaintextType oldPtType =
        cast<lwe::LWEPlaintextType>(encodeOp.getResult().getType());

    LLVM_DEBUG(llvm::dbgs()
               << "Forcing scaling factor of plaintext operand of "
               << op->getName()
               << " to match ciphertext operand. Old scaling factor="
               << lwe::getScalingFactorFromEncodingAttr(
                      oldPtType.getPlaintextSpace().getEncoding())
               << ", new="
               << lwe::getScalingFactorFromEncodingAttr(
                      ctType.getPlaintextSpace().getEncoding())
               << "\n");

    encodeOp.setEncodingAttr(ctType.getPlaintextSpace().getEncoding());
    lwe::LWEPlaintextType newPtType = lwe::LWEPlaintextType::get(
        op.getContext(), oldPtType.getApplicationData(),
        ctType.getPlaintextSpace());
    encodeOp.getResult().setType(newPtType);
  } else {
    // Not sure what to do here.
    op.emitError() << "Plaintext operand of " << op->getName()
                   << " is not directly encoded; cannot force scaling factor.";
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
          .Case<ckks::RescaleOp>(handleRescaleOp)
          .Case<ckks::AddPlainOp>(handleNonMulCtPtOp<ckks::AddPlainOp>)
          .Case<ckks::SubPlainOp>(handleNonMulCtPtOp<ckks::SubPlainOp>)
          .Case<ckks::MulPlainOp>(handleMulPlain)
          .Case<ckks::BootstrapOp>(handleBootstrap)
          // Some ops above implement InferTypeOpInterface, but need special
          // cases, so this must come after them.
          .Case<InferTypeOpInterface>(handleInferTypeOpInterface)
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
