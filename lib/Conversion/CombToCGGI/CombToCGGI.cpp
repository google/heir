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

class SecretTypeConverter : public TypeConverter {
 public:
  SecretTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // Convert secret types to LWE ciphertext types
    addConversion([ctx](secret::SecretType type) -> Type {
      auto intType = dyn_cast<IntegerType>(type.getValueType());
      assert(intType);
      return lwe::LWECiphertextType::get(
          ctx,
          lwe::UnspecifiedBitFieldEncodingAttr::get(ctx, intType.getWidth()),
          lwe::LWEParamsAttr());
    });
  }
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
    auto genericInputs = op.getInputs();
    for (int i = 0; i < opEntryBlock.getNumArguments(); i++) {
      rewriter.replaceAllUsesWith(opEntryBlock.getArgument(i),
                                  genericInputs[i]);
    }

    // In addition to lifting the plaintext arguments, we also lift the output
    // arguments to secrets. This is required for any truth tables that have
    // secret inputs.
    // For some reason, if this doesn't occur, the type conversion framework is
    // unable to update the uses of converted truth table results.
    rewriter.startRootUpdate(op);
    opEntryBlock.walk([&](comb::TruthTableOp op) {
      bool ciphertextArg =
          std::any_of(op.getOperands().begin(), op.getOperands().end(),
                      [&](const Value &val) {
                        return isa<secret::SecretType>(val.getType());
                      });
      if (ciphertextArg) {
        op->getResults()[0].setType(lwe::LWECiphertextType::get(
            getContext(),
            lwe::UnspecifiedBitFieldEncodingAttr::get(
                getContext(), op.getResult().getType().getWidth()),
            lwe::LWEParamsAttr()));
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
    bool ciphertextArg =
        std::any_of(adaptor.getOperands().begin(), adaptor.getOperands().end(),
                    [&](const Value &val) {
                      return isa<lwe::LWECiphertextType>(val.getType()) ||
                             isa<secret::SecretType>(val.getType());
                    });

    SmallVector<mlir::Value, 4> lutInputs;
    for (Value val : adaptor.getOperands()) {
      auto integerTy = dyn_cast<IntegerType>(val.getType());
      // If any of the arguments to the truth table are ciphertexts, we must
      // encode and trivially encrypt the plaintext integers arguments.
      if (ciphertextArg && integerTy) {
        auto encoding = lwe::UnspecifiedBitFieldEncodingAttr::get(
            ctx, integerTy.getWidth());
        auto ptxtTy = lwe::LWEPlaintextType::get(ctx, encoding);
        auto ctxtTy =
            lwe::LWECiphertextType::get(ctx, encoding, lwe::LWEParamsAttr());

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

struct CombToCGGI : public impl::CombToCGGIBase<CombToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    SecretTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    patterns.add<ConvertTruthTableOp>(typeConverter, context);
    target.addIllegalOp<TruthTableOp>();

    patterns.add<SecretGenericOpTypeConversion>(typeConverter,
                                                patterns.getContext());
    target.addDynamicallyLegalOp<secret::GenericOp>(
        [&](secret::GenericOp op) { return typeConverter.isLegal(op); });

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::comb
