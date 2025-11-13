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

using ckks::AddOp;
using ckks::AddPlainOp;
using ckks::BootstrapOp;
using ckks::LevelReduceOp;
using ckks::MulOp;
using ckks::MulPlainOp;
using ckks::RescaleOp;
using ckks::SubOp;
using ckks::SubPlainOp;
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
      xInput = MulPlainOp::create(b, xInput, encodedSplatRescale).getResult();
    }

    if (!shift.isZero()) {
      // unlike mul_plain, add_plain requires the encoded plaintext to have
      // the same scaling factor as the ciphertext.
      int64_t scalingFactor = lwe::getScalingFactorFromEncodingAttr(
          ctTy.getPlaintextSpace().getEncoding());
      auto encodedSplatShift = encodeSplattedCleartextUsingCtAndScalingFactor(
          b, ctTy, scalingFactor, shift);
      xInput = AddPlainOp::create(b, xInput, encodedSplatShift).getResult();
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
      auto rescaleOp = RescaleOp::create(b, moddedDownTy, finalOutput,
                                         ctTy.getCiphertextSpace().getRing());
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
      auto rescaleOp = RescaleOp::create(b, ctTy, finalOutput,
                                         ctTy.getCiphertextSpace().getRing());
      finalOutput = rescaleOp.getResult();
    }

    rewriter.replaceOp(op, finalOutput);
    return success();
  }

 private:
  std::string libraryTarget;
};

struct FixOperandsForRescale : public OpRewritePattern<RescaleOp> {
  using OpRewritePattern<RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RescaleOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Handling RescaleOp\n");
    // We re-infer the result type of the rescale op, and if the level drops to
    // zero we insert a bootstrap. This is necessary when lowering from Orion
    // to OpenFHE, because OpenFHE has different scaling behavior than Lattigo,
    // which Orion was written to target.
    lwe::LWECiphertextType inputType =
        cast<lwe::LWECiphertextType>(op.getInput().getType());
    FailureOr<lwe::LWECiphertextType> outputTypeResult =
        applyModReduce(inputType);
    Value input = op.getInput();
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
        return failure();
      }
      auto bootstrapOp =
          BootstrapOp::create(rewriter, op.getLoc(),
                              bootstrapResultTypeResult.value(), op.getInput());
      input = bootstrapOp.getResult();

      outputTypeResult = applyModReduce(bootstrapResultTypeResult.value());
      if (failed(outputTypeResult)) {
        return op.emitError()
               << "Failed to rescale even after inserting bootstrap\n";
      }
    }
    debugLevelAndScale(outputTypeResult.value());
    rewriter.replaceOpWithNewOp<RescaleOp>(
        op, outputTypeResult.value(), input,
        outputTypeResult.value().getCiphertextSpace().getRing());
    return success();
  }
};

struct FixOperandsForBootstrap : public OpRewritePattern<BootstrapOp> {
  using OpRewritePattern<BootstrapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BootstrapOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Handling BootstrapOp\n");
    // First, we need to find the maximum level in the modulus chain from the
    // ciphertext type.
    lwe::LWECiphertextType inputType =
        cast<lwe::LWECiphertextType>(op.getInput().getType());
    // sub 1 because the max level is the last index in the chain.
    int64_t maxLevel = inputType.getModulusChain().getElements().size() - 1;

    // Now we cheat a little bit: normally bootstrap itself would consume some
    // levels, which depends on the chosen backend. In our case, we're lowering
    // to library backends that handle this opaquely, so as long as we keep the
    // type system consistent at this step, we're fine.
    //
    // TODO(#1207): fix if this pass still matters when lowering to polynomial.
    FailureOr<lwe::LWECiphertextType> outputTypeResult =
        cloneAtLevel(inputType, maxLevel);
    debugLevelAndScale(outputTypeResult.value());
    rewriter.modifyOpInPlace(
        op, [&]() { op.getResult().setType(outputTypeResult.value()); });
    return success();
  }
};

struct FixOperandsForMulPlain : public OpRewritePattern<MulPlainOp> {
  using OpRewritePattern<MulPlainOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulPlainOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Handling MulPlainOp\n");
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
        inferMulOpPlaintextSpaceAttr(op.getContext(),
                                     ctType.getPlaintextSpace(),
                                     ptType.getPlaintextSpace()),
        ctType.getCiphertextSpace(), ctType.getKey(), ctType.getModulusChain());
    debugLevelAndScale(newCtType);
    rewriter.replaceOpWithNewOp<MulPlainOp>(op, newCtType, op.getLhs(),
                                            op.getRhs());
    return success();
  }
};

struct FixOperandsForMul : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Handling Mul op\n");
    lwe::LWECiphertextType lhsType =
        cast<lwe::LWECiphertextType>(op.getOperand(0).getType());
    lwe::LWECiphertextType rhsType =
        cast<lwe::LWECiphertextType>(op.getOperand(1).getType());

    // Mul ops may have differing scales, but not differing levels
    int64_t lhsLevel = lhsType.getModulusChain().getCurrent();
    int64_t rhsLevel = rhsType.getModulusChain().getCurrent();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (lhsLevel != rhsLevel) {
      LLVM_DEBUG(llvm::dbgs()
                 << "lhs level = " << lhsLevel << " != rhs level = " << rhsLevel
                 << ", applying level_reduce...\n");
      Value operandToReduce;
      int64_t levelsToDrop;
      if (lhsLevel < rhsLevel) {
        operandToReduce = op.getOperand(1);
        levelsToDrop = rhsLevel - lhsLevel;
      } else {
        operandToReduce = op.getOperand(0);
        levelsToDrop = lhsLevel - rhsLevel;
      }
      LLVM_DEBUG(llvm::dbgs() << "dropping " << levelsToDrop << " levels\n");

      auto levelReduceOp =
          LevelReduceOp::create(rewriter, op.getLoc(), operandToReduce,
                                rewriter.getI64IntegerAttr(levelsToDrop));
      debugLevelAndScale(levelReduceOp.getResult().getType(), "operand");

      if (lhsLevel > rhsLevel) {
        lhs = levelReduceOp.getResult();
      } else {
        rhs = levelReduceOp.getResult();
      }
    }

    rewriter.replaceOpWithNewOp<MulOp>(op, lhs, rhs);
    return success();
  }
};

// Ops like ckks.add and ckks.sub require both operands to have the same level
// and scaling factor.
template <typename CtCtOp>
struct FixOperandsForBinop : public OpRewritePattern<CtCtOp> {
  using OpRewritePattern<CtCtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CtCtOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Handling ct-ct op " << op->getName() << "\n");

    CtCtOp currentOp = op;

    // Determine if we need to reduce the level of one operand to match the
    // other, or rescale, or do both simultaneously.
    Value lhs = currentOp.getLhs();
    Value rhs = currentOp.getRhs();
    auto lhsType = cast<lwe::LWECiphertextType>(lhs.getType());
    auto rhsType = cast<lwe::LWECiphertextType>(rhs.getType());
    auto lhsLevel = lhsType.getModulusChain().getCurrent();
    auto rhsLevel = rhsType.getModulusChain().getCurrent();
    auto lhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
        lhsType.getPlaintextSpace().getEncoding());
    auto rhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
        rhsType.getPlaintextSpace().getEncoding());

    // Rescale to adjust scale and level simultaneously
    if (lhsLevel != rhsLevel && lhsScalingFactor != rhsScalingFactor) {
      LLVM_DEBUG(llvm::dbgs()
                 << "lhs level = " << lhsLevel << " != rhs level = " << rhsLevel
                 << ", and " << "lhs scale = " << lhsScalingFactor
                 << " != rhs scale = " << rhsScalingFactor
                 << ", applying rescale...\n");
      int64_t numRescales = 1;
      Value operandToRescale;
      int64_t logDefaultScale = getLogDefaultScale(op);
      if (lhsScalingFactor > rhsScalingFactor) {
        // This would break with high-precision scaling
        numRescales = (lhsScalingFactor - rhsScalingFactor) / logDefaultScale;
        operandToRescale = op.getLhs();
      } else {
        numRescales = (rhsScalingFactor - lhsScalingFactor) / logDefaultScale;
        operandToRescale = op.getRhs();
      }

      for (int64_t i = 0; i < numRescales; i++) {
        LLVM_DEBUG(llvm::dbgs() << "applying rescale " << (i + 1) << "/"
                                << numRescales << "\n");
        auto rescaleInputTy =
            cast<lwe::LWECiphertextType>(operandToRescale.getType());
        FailureOr<lwe::LWECiphertextType> ctTypeResult =
            applyModReduce(rescaleInputTy);
        if (failed(ctTypeResult)) {
          return op.emitError() << "Cannot rescale ciphertext type\n";
        }
        operandToRescale =
            RescaleOp::create(
                rewriter, op.getLoc(), ctTypeResult.value(), operandToRescale,
                ctTypeResult.value().getCiphertextSpace().getRing())
                .getResult();
      }

      if (lhsScalingFactor > rhsScalingFactor) {
        lhs = operandToRescale;
      } else {
        rhs = operandToRescale;
      }
      debugLevelAndScale(operandToRescale.getType(), "operand");

      lhsType = cast<lwe::LWECiphertextType>(lhs.getType());
      rhsType = cast<lwe::LWECiphertextType>(rhs.getType());
      lhsLevel = lhsType.getModulusChain().getCurrent();
      rhsLevel = rhsType.getModulusChain().getCurrent();
      lhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
          lhsType.getPlaintextSpace().getEncoding());
      rhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
          rhsType.getPlaintextSpace().getEncoding());
    }

    // After rescaling, now the levels may still mismatch and we need to
    // drop levels (without rescaling) to align.
    //
    // Example:
    //
    //  lhs level = 10 != rhs level = 8
    //  lhs scale = 80 != rhs scale = 120
    //
    // Above we rescale the rhs from 120 to 80, which drops the level to 7, then
    // we level reduce the lhs from 10 to 7.
    if (lhsLevel != rhsLevel) {
      LLVM_DEBUG(llvm::dbgs()
                 << "lhs level = " << lhsLevel << " != rhs level = " << rhsLevel
                 << ", applying level_reduce...\n");
      Value operandToReduce;
      int64_t levelsToDrop;
      if (lhsLevel < rhsLevel) {
        operandToReduce = rhs;
        levelsToDrop = rhsLevel - lhsLevel;
      } else {
        operandToReduce = lhs;
        levelsToDrop = lhsLevel - rhsLevel;
      }

      LLVM_DEBUG(llvm::dbgs() << "dropping " << levelsToDrop << " levels\n");
      auto levelReduceOp =
          LevelReduceOp::create(rewriter, op.getLoc(), operandToReduce,
                                rewriter.getI64IntegerAttr(levelsToDrop));
      debugLevelAndScale(levelReduceOp.getResult().getType(), "operand");

      if (lhsLevel > rhsLevel) {
        lhs = levelReduceOp.getResult();
      } else {
        rhs = levelReduceOp.getResult();
      }
      lhsType = cast<lwe::LWECiphertextType>(lhs.getType());
      rhsType = cast<lwe::LWECiphertextType>(rhs.getType());
      lhsLevel = lhsType.getModulusChain().getCurrent();
      rhsLevel = rhsType.getModulusChain().getCurrent();
      lhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
          lhsType.getPlaintextSpace().getEncoding());
      rhsScalingFactor = lwe::getScalingFactorFromEncodingAttr(
          rhsType.getPlaintextSpace().getEncoding());
    }

    if (lhsScalingFactor != rhsScalingFactor) {
      LLVM_DEBUG(llvm::dbgs() << "lhs scale = " << lhsScalingFactor
                              << " != rhs scale = " << rhsScalingFactor
                              << ", applying mul_plain rescale...\n");
      Value operandToRescale;
      int64_t targetScalingFactor;
      if (lhsScalingFactor < rhsScalingFactor) {
        operandToRescale = lhs;
        targetScalingFactor = rhsScalingFactor;
      } else {
        operandToRescale = rhs;
        targetScalingFactor = lhsScalingFactor;
      }

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      auto encodedSplatOne = encodeSplattedCleartextUsingCtAndScalingFactor(
          b, cast<lwe::LWECiphertextType>(operandToRescale.getType()),
          targetScalingFactor, APFloat(1.0));
      auto mulPlainOp =
          MulPlainOp::create(b, operandToRescale, encodedSplatOne);
      debugLevelAndScale(mulPlainOp.getResult().getType(), "operand");

      if (lhsScalingFactor < rhsScalingFactor) {
        lhs = mulPlainOp.getResult();
      } else {
        rhs = mulPlainOp.getResult();
      }
    }

    rewriter.replaceOpWithNewOp<CtCtOp>(op, lhs, rhs);
    return success();
  }
};

template <typename CtPtOp>
struct FixOperandsForCtPtBinop : public OpRewritePattern<CtPtOp> {
  using OpRewritePattern<CtPtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CtPtOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Handling ct-pt op " << op->getName() << "\n");

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

      encodeOp.setEncodingAttr(ctType.getPlaintextSpace().getEncoding());
      lwe::LWEPlaintextType newPtType = lwe::LWEPlaintextType::get(
          op.getContext(), oldPtType.getApplicationData(),
          ctType.getPlaintextSpace());
      encodeOp.getResult().setType(newPtType);
    } else {
      // Not sure what to do here.
      op.emitError()
          << "Plaintext operand of " << op->getName()
          << " is not directly encoded; cannot force scaling factor.";
    }
    debugLevelAndScale(ctType);

    Value rhs, lhs;
    if (isa<lwe::LWECiphertextType>(op.getLhs().getType())) {
      lhs = op.getLhs();
      rhs = plaintextOperand;
    } else {
      lhs = plaintextOperand;
      rhs = op.getRhs();
    }

    rewriter.replaceOpWithNewOp<CtPtOp>(op, lhs, rhs);
    return success();
  }
};

struct FixInferTypeOpInterface
    : public OpInterfaceRewritePattern<InferTypeOpInterface> {
  using OpInterfaceRewritePattern<
      InferTypeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(InferTypeOpInterface op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "Handling infer type op interface " << op->getName() << "\n");

    SmallVector<Type> inferredTypes;
    if (failed(op.inferReturnTypes(rewriter.getContext(), op.getLoc(),
                                   op->getOperands(), op->getAttrDictionary(),
                                   op->getPropertiesStorage(), op->getRegions(),
                                   inferredTypes))) {
      return op.emitError() << "Failed to infer return types";
    }

    OperationState state(op.getLoc(), op->getName(), op->getOperands(),
                         inferredTypes, op->getAttrs());
    Operation* newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

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
    RewritePatternSet cleanupPatterns(context);
    cleanupPatterns
        .add<FixOperandsForRescale, FixOperandsForBootstrap,
             FixOperandsForMulPlain, FixOperandsForMul,
             FixOperandsForBinop<AddOp>, FixOperandsForBinop<SubOp>,
             FixOperandsForCtPtBinop<AddOp>, FixOperandsForCtPtBinop<SubOp>>(
            context);
    cleanupPatterns.add<FixInferTypeOpInterface>(context, /*benefit=*/0);
    walkAndApplyPatterns(root, std::move(cleanupPatterns));

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
