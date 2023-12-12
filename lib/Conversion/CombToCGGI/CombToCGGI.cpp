#include "include/Conversion/CombToCGGI/CombToCGGI.h"

#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/Comb/IR/CombOps.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/LWE/IR/LWEOps.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Conversion/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::comb {

#define GEN_PASS_DEF_COMBTOCGGI
#include "include/Conversion/CombToCGGI/CombToCGGI.h.inc"

namespace {

bool isCiphertextOrSecret(Type type) {
  if (isa<secret::SecretType>(type) || isa<lwe::LWECiphertextType>(type)) {
    return true;
  }
  if (ShapedType shapedType = dyn_cast<ShapedType>(type)) {
    return isCiphertextOrSecret(shapedType.getElementType());
  }
  return false;
}

// equivalentMultiBitAndTensor checks whether the candidateMultiBit integer type
// is equivalent to the candidateTensor type.
// They are equivalent if the candidateTensor is a tensor of single bits with
// size equal to the number of bits of the candidateMultiBit.
bool equivalentMultiBitAndTensor(Type candidateMultiBit, Type candidateTensor) {
  if (auto multiBitTy = dyn_cast<IntegerType>(candidateMultiBit)) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(candidateTensor)) {
      auto tensorElt = dyn_cast<IntegerType>(tensorTy.getElementType());
      if (tensorElt && multiBitTy.getWidth() ==
                           tensorTy.getNumElements() * tensorElt.getWidth()) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

class SecretTypeConverter : public TypeConverter {
 public:
  SecretTypeConverter(MLIRContext *ctx, int minBitWidth)
      : minBitWidth(minBitWidth) {
    addConversion([](Type type) { return type; });

    // Convert secret types to LWE ciphertext types
    addConversion([ctx, this](secret::SecretType type) -> Type {
      return getLWECiphertextForInt(ctx, type.getValueType());
    });
  }

  Type getLWECiphertextForInt(MLIRContext *ctx, Type type) const {
    if (IntegerType intType = dyn_cast<IntegerType>(type)) {
      if (intType.getWidth() == 1) {
        return lwe::LWECiphertextType::get(
            ctx, lwe::UnspecifiedBitFieldEncodingAttr::get(ctx, minBitWidth),
            lwe::LWEParamsAttr());
      }
      return RankedTensorType::get(
          {intType.getWidth()},
          getLWECiphertextForInt(ctx, IntegerType::get(ctx, 1)));
    }
    ShapedType shapedType = dyn_cast<ShapedType>(type);
    assert(shapedType &&
           "expected shaped secret type for a non-integer secret");
    assert(isa<IntegerType>(shapedType.getElementType()) &&
           "expected integer element types for shaped secret types");
    return shapedType.cloneWith(
        shapedType.getShape(),
        getLWECiphertextForInt(ctx, shapedType.getElementType()));
  }

  int minBitWidth;
};

class SecretGenericOpTypeConversion
    : public OpConversionPattern<secret::GenericOp> {
 public:
  using OpConversionPattern<secret::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Block *originalBlock = op->getBlock();
    Block &opEntryBlock = op.getRegion().front();

    secret::YieldOp yieldOp =
        dyn_cast<secret::YieldOp>(op.getRegion().back().getTerminator());

    // Split the parent block of the generic op, so that we have a
    // clear insertion point for inlining.
    Block *newBlock = rewriter.splitBlock(originalBlock, Block::iterator(op));

    // mergeBlocks does not replace the original block values with the inputs to
    // secret.generic, so we manually replace them here. This lifts the internal
    // plaintext integer values within the secret.generic body to their original
    // secret values.
    auto genericInputs = adaptor.getInputs();
    for (unsigned i = 0; i < opEntryBlock.getNumArguments(); i++) {
      rewriter.replaceAllUsesWith(opEntryBlock.getArgument(i),
                                  genericInputs[i]);
    }

    // In addition to lifting the plaintext arguments, we also lift the output
    // arguments to secrets. This is required for any truth tables that have
    // secret inputs.
    // For some reason, if this doesn't occur, the type conversion framework is
    // unable to update the uses of converted truth table results.
    rewriter.startRootUpdate(op);
    const SecretTypeConverter *secretConverter =
        static_cast<const SecretTypeConverter *>(typeConverter);
    opEntryBlock.walk<WalkOrder::PreOrder>([&](Operation *op) {
      bool ciphertextArg =
          std::any_of(op->getOperands().begin(), op->getOperands().end(),
                      [&](const Value &val) {
                        return isCiphertextOrSecret(val.getType());
                      });
      if (ciphertextArg) {
        for (unsigned i = 0; i < op->getNumResults(); i++) {
          op->getResult(i).setType(secretConverter->getLWECiphertextForInt(
              getContext(), op->getResult(i).getType()));
        }
      }
    });

    rewriter.finalizeRootUpdate(op);

    // Inline the secret.generic internal region, moving all of the operations
    // to the parent region.
    rewriter.inlineRegionBefore(op.getRegion(), newBlock);
    rewriter.replaceOp(op, yieldOp->getOperands());
    rewriter.mergeBlocks(&opEntryBlock, originalBlock, genericInputs);
    rewriter.mergeBlocks(newBlock, originalBlock, {});

    rewriter.eraseOp(yieldOp);
    return success();
  }
};

// ConvertTruthTableOp converts op arguments to trivially encoded LWE
// ciphertexts when at least one argument is an LWE ciphertext.
struct ConvertTruthTableOp : public OpConversionPattern<TruthTableOp> {
  ConvertTruthTableOp(mlir::MLIRContext *context)
      : OpConversionPattern<TruthTableOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TruthTableOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3) {
      op->emitError() << "expected 3 truth table arguments to lower to CGGI";
    }

    MLIRContext *ctx = getContext();
    bool ciphertextArg = std::any_of(
        adaptor.getOperands().begin(), adaptor.getOperands().end(),
        [&](const Value &val) { return isCiphertextOrSecret(val.getType()); });

    SmallVector<mlir::Value, 4> lutInputs;
    const SecretTypeConverter *secretConverter =
        static_cast<const SecretTypeConverter *>(typeConverter);
    for (Value val : adaptor.getOperands()) {
      auto integerTy = dyn_cast<IntegerType>(val.getType());
      // If any of the arguments to the truth table are ciphertexts, we must
      // encode and trivially encrypt the plaintext integers arguments.
      if (ciphertextArg && integerTy) {
        assert(integerTy.getWidth() == 1 && "LUT inputs should be single-bit");
        auto ctxtTy = secretConverter->getLWECiphertextForInt(ctx, integerTy);
        auto encoding = cast<lwe::LWECiphertextType>(ctxtTy).getEncoding();
        auto ptxtTy = lwe::LWEPlaintextType::get(ctx, encoding);

        lutInputs.push_back(rewriter.create<lwe::TrivialEncryptOp>(
            op.getLoc(), ctxtTy,
            rewriter.create<lwe::EncodeOp>(op.getLoc(), ptxtTy, val, encoding),
            lwe::LWEParamsAttr()));
      } else {
        lutInputs.push_back(val);
      }
    }

    rewriter.replaceOp(op, rewriter.create<cggi::Lut3Op>(
                               op.getLoc(), lutInputs[0], lutInputs[1],
                               lutInputs[2], op.getLookupTable()));

    return success();
  }
};

// ConvertSecretCastOp removes secret.cast operations between multi-bit secret
// integers and tensors of single-bit secrets.
struct ConvertSecretCastOp : public OpConversionPattern<secret::CastOp> {
  ConvertSecretCastOp(mlir::MLIRContext *context)
      : OpConversionPattern<secret::CastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      secret::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // If this is a cast from secret<i8> to secret<tensor<8xi1>> or vice
    // versa, replace with the cast's input.
    auto inputTy =
        cast<secret::SecretType>(op.getInput().getType()).getValueType();
    auto outputTy =
        cast<secret::SecretType>(op.getOutput().getType()).getValueType();

    if (equivalentMultiBitAndTensor(inputTy, outputTy) ||
        equivalentMultiBitAndTensor(outputTy, inputTy)) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    return failure();
  }
};

struct CombToCGGI : public impl::CombToCGGIBase<CombToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    // TODO(https://github.com/google/heir/issues/250): The lutSize here is
    // fixed on the assumption that the comb dialect is using ternary LUTs.
    // Generalize lutSize by doing an analysis pass on the input combinational
    // operations and integers.
    int lutSize = 3;
    SecretTypeConverter typeConverter(context, lutSize);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    patterns.add<ConvertTruthTableOp, ConvertSecretCastOp,
                 SecretGenericOpTypeConversion>(typeConverter, context);
    target.addIllegalOp<TruthTableOp, secret::GenericOp, secret::CastOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::comb
