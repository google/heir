#include "lib/Dialect/Secret/Transforms/ForgetSecrets.h"

#include <cassert>
#include <utility>

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/TransformUtils.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

#define GEN_PASS_DEF_SECRETFORGETSECRETS
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::func::ReturnOp;

static Value materializeSource(OpBuilder& builder, Type type, ValueRange inputs,
                               Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (isa<SecretType>(inputType))
    // This suggests a bug with the dialect conversion infrastructure,
    // or else that sercrets have been improperly nested.
    llvm_unreachable(
        "Secret types should never be the input to a materializeSource.");

  return ConcealOp::create(builder, loc, inputs[0]);
}

static Value materializeTarget(OpBuilder& builder, Type type, ValueRange inputs,
                               Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (!isa<SecretType>(inputType))
    llvm_unreachable(
        "Non-secret types should never be the input to a materializeTarget.");

  return RevealOp::create(builder, loc, inputs[0]);
}

class ForgetSecretsTypeConverter : public TypeConverter {
 public:
  ForgetSecretsTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](SecretType secretType) -> Type {
      return secretType.getValueType();
    });

    addSourceMaterialization(materializeSource);
    addTargetMaterialization(materializeTarget);
  }
};

struct ConvertGeneric : public OpConversionPattern<GenericOp> {
  ConvertGeneric(mlir::MLIRContext* context)
      : OpConversionPattern<GenericOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    op.inlineInPlaceDroppingSecrets(rewriter, adaptor.getOperands());
    return success();
  }
};

struct ConvertCast : public OpConversionPattern<CastOp> {
  ConvertCast(mlir::MLIRContext* context)
      : OpConversionPattern<CastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto sourceType = adaptor.getInput().getType();
    auto targetType = getTypeConverter()->convertType(op.getResult().getType());

    if (sourceType == targetType) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    if (isa<MemRefType>(targetType) && isa<IntegerType>(sourceType)) {
      // Decompose the bits of the input into a memref of i1s.
      auto convertedValue = convertIntegerValueToMemrefOfBits(
          adaptor.getInput(), rewriter, op.getLoc());
      rewriter.replaceOp(op, convertedValue);
      return success();
    }

    if (isa<MemRefType>(sourceType) && isa<IntegerType>(targetType)) {
      // Reconstruct the integer type from a memref of its bits.
      auto convertedValue = convertMemrefOfBitsToInteger(
          adaptor.getInput(), targetType, rewriter, op.getLoc());
      rewriter.replaceOp(op, convertedValue);
      return success();
    }

    return op->emitOpError()
           << "SecretForgetSecrets does not yet support lowering a cast from "
           << sourceType << " to " << targetType
           << ", but if you see this, maybe it should support your use case "
              "and you can contribute the patch :)";
  }
};

struct ForgetSecrets : impl::SecretForgetSecretsBase<ForgetSecrets> {
  using SecretForgetSecretsBase::SecretForgetSecretsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* func = getOperation();
    ConversionTarget target(*context);
    ForgetSecretsTypeConverter typeConverter;

    target.addIllegalDialect<SecretDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
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
    patterns
        .add<ConvertGeneric, DropOp<ConcealOp>, DropOp<RevealOp>, ConvertCast>(
            typeConverter, context);
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(func, target, std::move(patterns),
                                      config))) {
      signalPassFailure();
    }

    // Clear any tensor_ext attributes from the func
    getOperation()->walk([&](FunctionOpInterface funcOp) {
      for (int i = 0; i < funcOp.getNumArguments(); ++i) {
        for (auto attr : funcOp.getArgAttrs(i)) {
          // the attr name is tensor_ext.foo, so just check for the prefix
          if (attr.getName().getValue().starts_with("tensor_ext")) {
            funcOp.removeArgAttr(i, attr.getName());
          }
        }
      }

      for (int i = 0; i < funcOp.getNumResults(); ++i) {
        for (auto attr : funcOp.getResultAttrs(i)) {
          if (attr.getName().getValue().starts_with("tensor_ext")) {
            funcOp.removeResultAttr(i, attr.getName());
          }
        }
      }
    });
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
