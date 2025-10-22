#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::orion {

#define GEN_PASS_DEF_ORIONTOCKKS
#include "lib/Dialect/Orion/Conversions/OrionToCKKS/OrionToCKKS.h.inc"

struct ConvertChebyshevOp : public OpConversionPattern<ChebyshevOp> {
  ConvertChebyshevOp(mlir::MLIRContext *context)
      : OpConversionPattern<ChebyshevOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChebyshevOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // FIXME: implement
    return failure();
  }
};

struct ConvertLinearTransformOp
    : public OpConversionPattern<LinearTransformOp> {
  ConvertLinearTransformOp(mlir::MLIRContext *context)
      : OpConversionPattern<LinearTransformOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // FIXME: implement
    return failure();
  }
};

struct OrionToCKKS : public impl::OrionToCKKSBase<OrionToCKKS> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<ckks::CKKSDialect>();
    target.addIllegalDialect<orion::OrionDialect>();
    patterns.add<ConvertChebyshevOp, ConvertLinearTransformOp>(context);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::orion
