#include "include/Conversion/CGGIToTfheRustBool/CGGIToTfheRustBool.h"

#include <utility>

#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/LWE/IR/LWEOps.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "lib/Conversion/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOTFHERUSTBOOL
#include "include/Conversion/CGGIToTfheRustBool/CGGIToTfheRustBool.h.inc"

class BoolPassTypeConverter : public TypeConverter {
 public:
  BoolPassTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      return tfhe_rust_bool::EncryptedBoolType::get(ctx);
    });
    addConversion([this](ShapedType type) -> Type {
      return type.cloneWith(type.getShape(),
                            this->convertType(type.getElementType()));
    });
  }
};

// /// Returns true if the func's body contains any CGGI ops.
bool containsCGGIOpsBool(func::FuncOp func) {
  auto walkResult = func.walk([&](Operation *op) {
    if (llvm::isa<cggi::CGGIDialect>(op->getDialect()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

/// Returns the Value corresponding to a server key in the FuncOp containing
/// this op.
FailureOr<Value> getContextualBoolServerKey(Operation *op) {
  Value serverKey = op->getParentOfType<func::FuncOp>()
                        .getBody()
                        .getBlocks()
                        .front()
                        .getArguments()
                        .front();
  if (!serverKey.getType().isa<tfhe_rust_bool::ServerKeyType>()) {
    return op->emitOpError()
           << "Found CGGI op in a function without a server "
              "key argument. Did the AddBoolServerKeyArg pattern fail to run?";
  }
  return serverKey;
}

template <class Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

/// Convert a func by adding a server key argument. Converted ops in other
/// patterns need a server key SSA value available, so this pattern needs a
/// higher benefit.
struct AddBoolServerKeyArg : public OpConversionPattern<func::FuncOp> {
  AddBoolServerKeyArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsCGGIOpsBool(op)) {
      return failure();
    }

    auto serverKeyType = tfhe_rust_bool::ServerKeyType::get(getContext());
    FunctionType originalType = op.getFunctionType();
    llvm::SmallVector<Type, 4> newTypes;
    newTypes.reserve(originalType.getNumInputs() + 1);
    newTypes.push_back(serverKeyType);
    for (auto t : originalType.getInputs()) {
      newTypes.push_back(t);
    }
    auto newFuncType =
        FunctionType::get(getContext(), newTypes, originalType.getResults());
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(newFuncType);

      // In addition to updating the type signature, we need to update the
      // entry block's arguments to match the type signature
      Block &block = op.getBody().getBlocks().front();
      block.insertArgument(&block.getArguments().front(), serverKeyType,
                           op.getLoc());
    });

    return success();
  }
};

struct ConvertBoolAndOp : public OpConversionPattern<cggi::AndOp> {
  ConvertBoolAndOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::AndOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::AndOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualBoolServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(op, b.create<tfhe_rust_bool::AndOp>(
                               serverKey, adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ConvertBoolOrOp : public OpConversionPattern<cggi::OrOp> {
  ConvertBoolOrOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::OrOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::OrOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualBoolServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(op, b.create<tfhe_rust_bool::OrOp>(
                               serverKey, adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ConvertBoolXorOp : public OpConversionPattern<cggi::XorOp> {
  ConvertBoolXorOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::XorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::XorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualBoolServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(op, b.create<tfhe_rust_bool::XorOp>(
                               serverKey, adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ConvertBoolNotOp : public OpConversionPattern<cggi::NotOp> {
  ConvertBoolNotOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::NotOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::NotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualBoolServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(op, b.create<tfhe_rust_bool::NotOp>(
                               serverKey, adaptor.getInput()));
    return success();
  }
};

struct ConvertBoolTrivialEncryptOp
    : public OpConversionPattern<lwe::TrivialEncryptOp> {
  ConvertBoolTrivialEncryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::TrivialEncryptOp>(context, /*benefit=*/2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::TrivialEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualBoolServerKey(op.getOperation());
    if (failed(result)) return result;

    Value serverKey = result.value();
    lwe::EncodeOp encodeOp = op.getInput().getDefiningOp<lwe::EncodeOp>();
    if (!encodeOp) {
      return op.emitError() << "Expected input to TrivialEncrypt to be the "
                               "result of an EncodeOp, but it was "
                            << op.getInput().getDefiningOp()->getName();
    }
    auto outputType = tfhe_rust_bool::EncryptedBoolType::get(getContext());
    ;
    auto createTrivialOp = rewriter.create<tfhe_rust_bool::CreateTrivialOp>(
        op.getLoc(), outputType, serverKey, encodeOp.getPlaintext());
    rewriter.replaceOp(op, createTrivialOp);
    return success();
  }
};

struct ConvertBoolEncodeOp : public OpConversionPattern<lwe::EncodeOp> {
  ConvertBoolEncodeOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::EncodeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::EncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class CGGIToTfheRustBool
    : public impl::CGGIToTfheRustBoolBase<CGGIToTfheRustBool> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *op = getOperation();

    BoolPassTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addLegalDialect<tfhe_rust_bool::TfheRustBoolDialect>();
    target.addIllegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();

    // FuncOp is marked legal by the default structural conversion patterns
    // helper, just based on type conversion. We need more, but because the
    // addDynamicallyLegalOp is a set-based method, we can add this after
    // calling addStructuralConversionPatterns and it will overwrite the
    // legality condition set in that function.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasServerKeyArg = op.getFunctionType().getNumInputs() > 0 &&
                             op.getFunctionType()
                                 .getInputs()
                                 .begin()
                                 ->isa<tfhe_rust_bool::ServerKeyType>();
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsCGGIOpsBool(op) || hasServerKeyArg);
    });
    target.addDynamicallyLegalOp<memref::AllocOp, memref::DeallocOp,
                                 memref::StoreOp, memref::LoadOp,
                                 memref::SubViewOp, memref::CopyOp,
                                 tensor::FromElementsOp, tensor::ExtractOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });

    // FIXME: still need to update callers to insert the new server key arg, if
    // needed and possible.
    patterns.add<
        AddBoolServerKeyArg, ConvertBoolAndOp, ConvertBoolEncodeOp, ConvertBoolOrOp,
        ConvertBoolTrivialEncryptOp, ConvertBoolXorOp, ConvertBoolNotOp,
        GenericOpPattern<memref::AllocOp>, GenericOpPattern<memref::DeallocOp>,
        GenericOpPattern<memref::StoreOp>, GenericOpPattern<memref::LoadOp>,
        GenericOpPattern<memref::SubViewOp>, GenericOpPattern<memref::CopyOp>,
        GenericOpPattern<tensor::FromElementsOp>,
        GenericOpPattern<tensor::ExtractOp>>(typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
