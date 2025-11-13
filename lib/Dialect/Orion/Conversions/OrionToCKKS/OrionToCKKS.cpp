#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/IRMaterializingVisitor.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Utils/Polynomial/ChebyshevPatersonStockmeyer.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Verifier.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "orion-to-ckks"

namespace mlir::heir::orion {

using kernel::ArithmeticDagNode;
using kernel::implementHaleviShoup;
using kernel::SSAValue;

#define GEN_PASS_DEF_ORIONTOCKKS
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

void debugLevelAndScale(Type ctTy, const std::string& type = "output") {
  auto ty = cast<lwe::LWECiphertextType>(ctTy);
  int64_t scale = lwe::getScalingFactorFromEncodingAttr(
      ty.getPlaintextSpace().getEncoding());
  int64_t level = ty.getModulusChain().getCurrent();
  LLVM_DEBUG(llvm::dbgs() << type << " level=" << level << ", scale=" << scale
                          << "\n");
}

lwe::LWEPlaintextType getPlaintextTypeFromCtTypeAndScalingFactor(
    lwe::LWECiphertextType ctTy, int64_t scalingFactor) {
  MLIRContext* ctx = ctTy.getContext();
  auto encodingAttr =
      lwe::InverseCanonicalEncodingAttr::get(ctx, scalingFactor);
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

int64_t getLogDefaultScale(Operation* op) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  auto ckksSchemeParamAttr = mlir::dyn_cast<ckks::SchemeParamAttr>(
      moduleOp->getAttr(ckks::CKKSDialect::kSchemeParamAttrName));
  return ckksSchemeParamAttr.getLogDefaultScale();
}

struct ConvertChebyshevOp : public OpRewritePattern<ChebyshevOp> {
  ConvertChebyshevOp(MLIRContext* context, std::string libraryTarget)
      : OpRewritePattern<ChebyshevOp>(context),
        libraryTarget(std::move(libraryTarget)) {}

  LogicalResult matchAndRewrite(ChebyshevOp op,
                                PatternRewriter& rewriter) const override {
    int64_t logDefaultScale = getLogDefaultScale(op);
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
          b, ctTy, logDefaultScale, rescale);
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

    bool rescaleAfterCtPtMul = (libraryTarget == "openfhe");
    IRMaterializingVisitor visitor(
        b, getPlaintextTypeFromCtTypeAndScalingFactor(ctTy, logDefaultScale),
        rescaleAfterCtPtMul, logDefaultScale);
    Value finalOutput = implementedKernel->visit(visitor);

    // ct-pt muls in the kernel didn't rescale, so rescale at the very end
    if (!rescaleAfterCtPtMul) {
      lwe::LWECiphertextType ctTy =
          cast<lwe::LWECiphertextType>(finalOutput.getType());
      FailureOr<lwe::LWECiphertextType> ctTypeResult = applyModReduce(ctTy);
      if (failed(ctTypeResult)) {
        emitError(op.getLoc()) << "Cannot rescale ciphertext type";
        return failure();
      }
      auto moddedDownTy = ctTypeResult.value();
      auto rescaleOp = ckks::RescaleOp::create(
          b, moddedDownTy, finalOutput, ctTy.getCiphertextSpace().getRing());
      finalOutput = rescaleOp.getResult();
    }

    rewriter.replaceOp(op, finalOutput);
    return success();
  }

 private:
  std::string libraryTarget;
};

struct ConvertLinearTransformOp : public OpRewritePattern<LinearTransformOp> {
  ConvertLinearTransformOp(MLIRContext* context, std::string libraryTarget)
      : OpRewritePattern<LinearTransformOp>(context),
        libraryTarget(std::move(libraryTarget)) {}

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

    bool rescaleAfterCtPtMul = (libraryTarget == "openfhe");
    int64_t logDefaultScale = getLogDefaultScale(op);
    lwe::LWECiphertextType ctTy = op.getInput().getType();
    IRMaterializingVisitor visitor(
        b, getPlaintextTypeFromCtTypeAndScalingFactor(ctTy, logDefaultScale),
        rescaleAfterCtPtMul, logDefaultScale);
    Value finalOutput = implementedKernel->visit(visitor);

    // ct-pt muls in the kernel didn't rescale, so rescale at the very end
    if (!rescaleAfterCtPtMul) {
      auto rescaleOp = ckks::RescaleOp::create(
          b, ctTy, finalOutput, ctTy.getCiphertextSpace().getRing());
      finalOutput = rescaleOp.getResult();
    }

    rewriter.replaceOp(op, finalOutput);
    return success();
  }

 private:
  std::string libraryTarget;
};

WalkResult handleInferTypeOpInterface(InferTypeOpInterface op) {
  LLVM_DEBUG(llvm::dbgs() << "Handling InferTypeOpInterface: " << op->getName()
                          << "\n");
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
      debugLevelAndScale(result.getType());
      result.setType(expectedType);
    }
  }
  return WalkResult::advance();
}

WalkResult handleRescaleOp(ckks::RescaleOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Handling RescaleOp\n");
  lwe::LWECiphertextType inputType =
      cast<lwe::LWECiphertextType>(op.getInput().getType());
  FailureOr<lwe::LWECiphertextType> outputTypeResult =
      applyModReduce(inputType);
  if (failed(outputTypeResult)) {
    op.emitError()
        << "Cannot drop one limb from ciphertext type, inserting bootstrap\n";
    int64_t maxLevel = inputType.getModulusChain().getElements().size() - 1;

    // Now we cheat a little bit: normally bootstrap itself would consume
    // some levels, which depends on the chosen backend. In our case, we're
    // lowering to library backends that handle this opaquely.
    //
    // TODO(#1207): fix if this pass still matters when lowering to
    // polynomial.
    FailureOr<lwe::LWECiphertextType> bootstrapResultTypeResult =
        cloneAtLevel(inputType, maxLevel);
    if (failed(bootstrapResultTypeResult)) {
      op.emitError() << "Failed to insert bootstrap";
      return WalkResult::interrupt();
    }
    ImplicitLocOpBuilder builder(op.getLoc(), op->getContext());
    builder.setInsertionPoint(op);
    auto bootstrapOp = ckks::BootstrapOp::create(
        builder, bootstrapResultTypeResult.value(), op.getInput());
    op.setOperand(bootstrapOp.getResult());

    outputTypeResult = applyModReduce(bootstrapResultTypeResult.value());
    if (failed(outputTypeResult)) {
      op.emitError() << "Failed to rescale even after inserting bootstrap\n";
      return WalkResult::interrupt();
    }
  }
  debugLevelAndScale(outputTypeResult.value());
  op.getResult().setType(outputTypeResult.value());
  op.setToRingAttr(outputTypeResult.value().getCiphertextSpace().getRing());
  return WalkResult::advance();
}

WalkResult handleBootstrap(ckks::BootstrapOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Handling BootstrapOp\n");
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
  debugLevelAndScale(outputTypeResult.value());
  op.getResult().setType(outputTypeResult.value());
  return WalkResult::advance();
}

template <typename CtPtOp>
WalkResult handleNonMulCtPtOp(CtPtOp op) {
  lwe::LWECiphertextType ctType;
  LLVM_DEBUG(llvm::dbgs() << "Handling CtPt op: " << op->getName() << "\n");
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

  debugLevelAndScale(ctType);
  op.getResult().setType(ctType);
  return WalkResult::advance();
}

WalkResult handleMulPlain(ckks::MulPlainOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Handling MulPlain op\n");
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
  debugLevelAndScale(newCtType);
  op.getResult().setType(newCtType);
  return WalkResult::advance();
}

// Handle a ciphertext-ciphertext operation by first ensuring that both
// operands are at the same level and have the same scaling factor. If both
// mismatch, we can insert a rescale op that drops an RNS limb and adjusts the
// larger scaling factor to the match the smaller.
//
// If only the levels mismatch, we insert a level_reduce op on the higher-level
// operand. If only the scaling factors mismatch, we insert a mul_plain op with
// an encoded splat to raise the smaller scaling factor to match the larger.
template <typename CtCtOp>
WalkResult handleCtCtOp(CtCtOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Handling ct-ct op: " << op->getName() << "\n");
  lwe::LWECiphertextType lhsType =
      cast<lwe::LWECiphertextType>(op.getLhs().getType());
  lwe::LWECiphertextType rhsType =
      cast<lwe::LWECiphertextType>(op.getRhs().getType());
  ImplicitLocOpBuilder b(op.getLoc(), op->getContext());
  b.setInsertionPoint(op);

  // Determine if we need to reduce the level of one operand to match the
  // other, or rescale, or do both simultaneously.
  int64_t lhsLevel = lhsType.getModulusChain().getCurrent();
  int64_t rhsLevel = rhsType.getModulusChain().getCurrent();
  int64_t lhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
      lhsType.getPlaintextSpace().getEncoding());
  int64_t rhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
      rhsType.getPlaintextSpace().getEncoding());

  if (lhsLevel != rhsLevel && lhsScalingFactor != rhsScalingFactor) {
    LLVM_DEBUG(llvm::dbgs()
               << "lhs level = " << lhsLevel << " != rhs level = " << rhsLevel
               << ", and " << "lhs scale = " << lhsScalingFactor
               << " != rhs scale = " << rhsScalingFactor
               << ", applying rescale...\n");
    Value operandToRescale;
    if (lhsScalingFactor >= rhsScalingFactor) {
      operandToRescale = op.getLhs();
    } else {
      operandToRescale = op.getRhs();
    }

    auto rescaleInputTy =
        cast<lwe::LWECiphertextType>(operandToRescale.getType());
    FailureOr<lwe::LWECiphertextType> ctTypeResult =
        applyModReduce(rescaleInputTy);
    if (failed(ctTypeResult)) {
      emitError(op.getLoc()) << "Cannot rescale ciphertext type\n";
      return WalkResult::interrupt();
    }
    auto ctType = ctTypeResult.value();
    auto rescaleOp = ckks::RescaleOp::create(
        b, ctType, operandToRescale, ctType.getCiphertextSpace().getRing());
    int64_t operandIndex = (lhsScalingFactor > rhsScalingFactor) ? 1 : 0;
    debugLevelAndScale(rescaleOp.getResult().getType(), "operand");
    op->setOperand(operandIndex, rescaleOp.getResult());
    return handleInferTypeOpInterface(op);
  }

  if (lhsLevel != rhsLevel) {
    LLVM_DEBUG(llvm::dbgs() << "lhs level = " << lhsLevel << " != rhs level = "
                            << rhsLevel << ", applying level_reduce...\n");
    Value operandToReduce;
    int64_t levelsToDrop;
    if (lhsLevel < rhsLevel) {
      operandToReduce = op.getRhs();
      levelsToDrop = rhsLevel - lhsLevel;
    } else {
      operandToReduce = op.getLhs();
      levelsToDrop = lhsLevel - rhsLevel;
    }

    auto levelReduceOp = ckks::LevelReduceOp::create(
        b, operandToReduce, b.getI64IntegerAttr(levelsToDrop));
    int64_t operandIndex = (lhsLevel < rhsLevel) ? 1 : 0;
    debugLevelAndScale(levelReduceOp.getResult().getType(), "operand");
    op->setOperand(operandIndex, levelReduceOp.getResult());
    return handleInferTypeOpInterface(op);
  }

  if (lhsScalingFactor != rhsScalingFactor) {
    LLVM_DEBUG(llvm::dbgs() << "lhs scale = " << lhsScalingFactor
                            << " != rhs scale = " << rhsScalingFactor
                            << ", applying mul_plain rescale...\n");
    Value operandToRescale;
    int64_t targetScalingFactor;
    if (lhsScalingFactor < rhsScalingFactor) {
      operandToRescale = op.getLhs();
      targetScalingFactor = rhsScalingFactor;
    } else {
      operandToRescale = op.getRhs();
      targetScalingFactor = lhsScalingFactor;
    }

    auto encodedSplatOne = encodeSplattedCleartextUsingCtAndScalingFactor(
        b, cast<lwe::LWECiphertextType>(operandToRescale.getType()),
        targetScalingFactor, APFloat(1.0));
    auto mulPlainOp =
        ckks::MulPlainOp::create(b, operandToRescale, encodedSplatOne);
    int64_t operandIndex = (lhsScalingFactor < rhsScalingFactor) ? 0 : 1;
    debugLevelAndScale(mulPlainOp.getResult().getType(), "operand");
    op->setOperand(operandIndex, mulPlainOp.getResult());
    return handleInferTypeOpInterface(op);
  }

  return handleInferTypeOpInterface(op);
}

WalkResult handleMul(ckks::MulOp op) {
  ImplicitLocOpBuilder b(op.getLoc(), op->getContext());
  b.setInsertionPoint(op);
  LLVM_DEBUG(llvm::dbgs() << "Handling Mul op\n");
  lwe::LWECiphertextType lhsType =
      cast<lwe::LWECiphertextType>(op.getLhs().getType());
  lwe::LWECiphertextType rhsType =
      cast<lwe::LWECiphertextType>(op.getRhs().getType());

  // Mul ops may have differing scales, but not differing levels
  int64_t lhsLevel = lhsType.getModulusChain().getCurrent();
  int64_t rhsLevel = rhsType.getModulusChain().getCurrent();

  if (lhsLevel != rhsLevel) {
    LLVM_DEBUG(llvm::dbgs() << "lhs level = " << lhsLevel << " != rhs level = "
                            << rhsLevel << ", applying level_reduce...\n");
    Value operandToReduce;
    int64_t levelsToDrop;
    if (lhsLevel < rhsLevel) {
      operandToReduce = op.getRhs();
      levelsToDrop = rhsLevel - lhsLevel;
    } else {
      operandToReduce = op.getLhs();
      levelsToDrop = lhsLevel - rhsLevel;
    }
    LLVM_DEBUG(llvm::dbgs() << "dropping " << levelsToDrop << " levels\n");

    auto levelReduceOp = ckks::LevelReduceOp::create(
        b, operandToReduce, b.getI64IntegerAttr(levelsToDrop));
    int64_t operandIndex = (lhsLevel < rhsLevel) ? 1 : 0;
    debugLevelAndScale(levelReduceOp.getResult().getType(), "operand");
    op->setOperand(operandIndex, levelReduceOp.getResult());
  }

  return handleInferTypeOpInterface(op);
}

struct OrionToCKKS : public impl::OrionToCKKSBase<OrionToCKKS> {
  using OrionToCKKSBase::OrionToCKKSBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp root = cast<ModuleOp>(getOperation());

    patterns.add<ConvertChebyshevOp, ConvertLinearTransformOp>(context,
                                                               libraryTarget);
    walkAndApplyPatterns(root, std::move(patterns));

    LLVM_DEBUG({
      (void)verify(root);
      llvm::dbgs() << "After lowering Chebyshev and LinearTransform "
                      "ops, but before repropagating types:\n";
      root.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // At this step, the types are wrong and need to be re-propagated In
    // particular, mul and mul_plain ops are followed by a rescale, and while
    // the result type drops a limb, the downstream ops are not updated to
    // match. Similarly, `mul` ops may need to have some arguments modded down
    // to match the other argument.
    WalkResult walkResult = root->walk([&](Operation* op) {
      if (op->hasTrait<OpTrait::SameOperandsAndResultType>()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping operation: " << op->getName()
                   << " because it has same operands and results\n");
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
          .Case<ckks::AddOp>(handleCtCtOp<ckks::AddOp>)
          .Case<ckks::SubOp>(handleCtCtOp<ckks::SubOp>)
          .Case<ckks::MulOp>(handleMul)
          .Case<ckks::BootstrapOp>(handleBootstrap)
          // Some ops above implement InferTypeOpInterface, but need special
          // cases, so this must come after them.
          .Case<InferTypeOpInterface>(handleInferTypeOpInterface)
          .Default([&](Operation* op) {
            if (llvm::none_of(op->getResults(), [](auto result) {
                  return isa<lwe::LWECiphertextType>(result.getType());
                })) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Skipping operation: " << op->getName()
                         << " because it has no ciphertext types\n");
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
