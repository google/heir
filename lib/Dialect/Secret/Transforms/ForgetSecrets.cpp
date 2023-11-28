#include "include/Dialect/Secret/Transforms/ForgetSecrets.h"

#include "include/Dialect/Secret/IR/SecretOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETFORGETSECRETS
#include "include/Dialect/Secret/Transforms/Passes.h.inc"

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::func::ReturnOp;

static Value materializeSource(OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (isa<SecretType>(inputType))
    // This suggests a bug with the dialect conversion infrastructure,
    // or else that sercrets have been improperly nested.
    llvm_unreachable(
        "Secret types should never be the input to a materializeSource.");

  return builder.create<ConcealOp>(loc, inputs[0]);
}

static Value materializeTarget(OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (!isa<SecretType>(inputType))
    llvm_unreachable(
        "Non-secret types should never be the input to a materializeTarget.");

  return builder.create<RevealOp>(loc, inputs[0]);
}

class ForgetSecretsTypeConverter : public TypeConverter {
 public:
  ForgetSecretsTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](SecretType secretType) -> Type {
      return secretType.getValueType();
    });

    addSourceMaterialization(materializeSource);
    addArgumentMaterialization(materializeSource);
    addTargetMaterialization(materializeTarget);
  }
};

struct ConvertGeneric : public OpConversionPattern<GenericOp> {
  ConvertGeneric(mlir::MLIRContext *context)
      : OpConversionPattern<GenericOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Block *originalBlock = op->getBlock();
    Block &opEntryBlock = op.getRegion().front();
    YieldOp yieldOp = dyn_cast<YieldOp>(op.getRegion().back().getTerminator());

    // Inline the op's (unique) block, including the yield op. This also
    // requires splitting the parent block of the generic op, so that we have a
    // clear insertion point for inlining.
    Block *newBlock = rewriter.splitBlock(originalBlock, Block::iterator(op));
    rewriter.inlineRegionBefore(op.getRegion(), newBlock);

    // Now that op's region is inlined, the operands of its YieldOp are mapped
    // to the materialized target values. Therefore, we can replace the op's
    // uses with those of its YieldOp's operands.
    rewriter.replaceOp(op, yieldOp->getOperands());

    // No need for these intermediate blocks, merge them into 1.
    rewriter.mergeBlocks(&opEntryBlock, originalBlock, adaptor.getOperands());
    rewriter.mergeBlocks(newBlock, originalBlock, {});

    rewriter.eraseOp(yieldOp);
    return success();
  }
};

struct ConvertConceal : public OpConversionPattern<ConcealOp> {
  ConvertConceal(mlir::MLIRContext *context)
      : OpConversionPattern<ConcealOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConcealOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getOutput().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOp(op, adaptor.getCleartext());
    return success();
  }
};

struct ConvertReveal : public OpConversionPattern<RevealOp> {
  ConvertReveal(mlir::MLIRContext *context)
      : OpConversionPattern<RevealOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RevealOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getCleartext().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct ForgetSecrets : impl::SecretForgetSecretsBase<ForgetSecrets> {
  using SecretForgetSecretsBase::SecretForgetSecretsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *func = getOperation();
    ConversionTarget target(*context);
    ForgetSecretsTypeConverter typeConverter;

    target.addIllegalDialect<SecretDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(context);
    patterns.add<ConvertGeneric, ConvertConceal, ConvertReveal>(typeConverter,
                                                                context);
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
