#include "include/Conversion/PolyToStandard/PolyToStandard.h"

#include "include/Dialect/Poly/IR/PolyOps.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "include/Conversion/PolyToStandard/PolyToStandard.h.inc"

class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](PolyType type) -> Type {
      RingAttr attr = type.getRing();
      uint32_t idealDegree = attr.ideal().getDegree();
      IntegerType elementTy =
          IntegerType::get(ctx, attr.coefficientModulus().getBitWidth(),
                           IntegerType::SignednessSemantics::Unsigned);
      return RankedTensorType::get({idealDegree}, elementTy, attr);
    });

    // We don't include any custom materialization ops because this lowering is
    // all done in a single pass. The dialect conversion framework works by
    // resolving intermediate (mid-pass) type conflicts by inserting
    // unrealized_conversion_cast ops, and only converting those to custom
    // materializations if they persist at the end of the pass. In our case,
    // we'd only need to use custom materializations if we split this lowering
    // across multiple passes.
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(https://github.com/google/heir/issues/104): implement
    return failure();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(mlir::MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(https://github.com/google/heir/issues/104): implement
    return success();
  }
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ConversionTarget target(*context);
    PolyToStandardTypeConverter typeConverter(context);

    // target.addIllegalDialect<PolyDialect>();
    target.addIllegalOp<FromTensorOp>();
    // target.addIllegalOp<AddOp>();
    // target.addIllegalOp<MulOp>();

    // TODO(https://github.com/google/heir/issues/105): add upstream type
    // converters for func/call/return

    RewritePatternSet patterns(context);
    patterns.add<ConvertFromTensor>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace poly
}  // namespace heir
}  // namespace mlir
