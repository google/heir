#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRustBool/CGGIToTfheRustBool.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolAttributes.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolEnums.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolOps.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "cggi-to-tfhe-rust-bool"

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOTFHERUSTBOOL
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRustBool/CGGIToTfheRustBool.h.inc"

class CGGIToTfheRustBoolTypeConverter : public TypeConverter {
 public:
  CGGIToTfheRustBoolTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::NewLWECiphertextType type) -> Type {
      return tfhe_rust_bool::EncryptedBoolType::get(ctx);
    });
    addConversion([this](ShapedType type) -> Type {
      return type.cloneWith(type.getShape(),
                            this->convertType(type.getElementType()));
    });
  }
};

/// Returns the Value corresponding to a server key in the FuncOp containing
/// this op.
FailureOr<Value> getContextualBoolServerKey(Operation *op) {
  Value serverKey = op->getParentOfType<func::FuncOp>()
                        .getBody()
                        .getBlocks()
                        .front()
                        .getArguments()
                        .front();
  if (!serverKey.getType().hasTrait<tfhe_rust_bool::ServerKeyTrait>()) {
    return op->emitOpError()
           << "Found CGGI op in a function without a server "
              "key argument. Did the AddBoolServerKeyArg pattern fail to run?";
  }
  return serverKey;
}

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
    if (!containsDialects<lwe::LWEDialect, cggi::CGGIDialect>(op)) {
      return failure();
    }

    Type serverKeyType = tfhe_rust_bool::ServerKeyType::get(getContext());
    auto containsPacked =
        op.walk([](cggi::PackedOp op) { return WalkResult::interrupt(); });
    if (containsPacked.wasInterrupted()) {
      serverKeyType = tfhe_rust_bool::PackedServerKeyType::get(getContext());
    }

    rewriter.modifyOpInPlace(op, [&] {
      (void)op.insertArgument(0, serverKeyType, nullptr, op.getLoc());
    });
    return success();
  }
};

template <typename BinOp, typename TfheRustBoolBinOp>
struct ConvertCGGITRBBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualBoolServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(op, b.create<TfheRustBoolBinOp>(
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

    rewriter.replaceOp(
        op, b.create<tfhe_rust_bool::NotOp>(serverKey, adaptor.getInput()));
    return success();
  }
};

struct ConvertPackedOp : public OpConversionPattern<cggi::PackedOp> {
  ConvertPackedOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::PackedOp>(context, /*benefit=*/1) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::PackedOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualBoolServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();
    auto *context = getContext();
    SmallVector<tfhe_rust_bool::TfheRustBoolGateEnumAttr, 4>
        vectorizedGateOperands;

    auto cggiGates = op.getGates().getGates();

    for (auto gate : cggiGates) {
      auto tfheGate = tfhe_rust_bool::symbolizeTfheRustBoolGateEnum(
          static_cast<uint32_t>(gate.getValue()));
      vectorizedGateOperands.push_back(
          tfhe_rust_bool::TfheRustBoolGateEnumAttr::get(context,
                                                        tfheGate.value()));
    }

    auto oplist = tfhe_rust_bool::TfheRustBoolGatesAttr::get(
        context, vectorizedGateOperands);

    auto outputType = adaptor.getLhs().getType();

    rewriter.replaceOp(op, b.create<tfhe_rust_bool::PackedOp>(
                               outputType, serverKey, oplist, adaptor.getLhs(),
                               adaptor.getRhs()));
    return success();
  }
};

struct ConvertBoolTrivialEncryptOp
    : public OpConversionPattern<lwe::TrivialEncryptOp> {
  ConvertBoolTrivialEncryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::TrivialEncryptOp>(context, /*benefit=*/1) {}

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

    auto createTrivialOp = rewriter.create<tfhe_rust_bool::CreateTrivialOp>(
        op.getLoc(), outputType, serverKey, encodeOp.getInput());
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

    CGGIToTfheRustBoolTypeConverter typeConverter(context);
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
      Type firstFuncType = *op.getFunctionType().getInputs().begin();
      bool hasServerKeyArg =
          op.getFunctionType().getNumInputs() > 0 &&
          firstFuncType.hasTrait<tfhe_rust_bool::ServerKeyTrait>();

      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsDialects<lwe::LWEDialect, cggi::CGGIDialect>(op) ||
              hasServerKeyArg);
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
    patterns.add<AddBoolServerKeyArg,
                 ConvertCGGITRBBinOp<cggi::AndOp, tfhe_rust_bool::AndOp>,
                 ConvertCGGITRBBinOp<cggi::NandOp, tfhe_rust_bool::NandOp>,
                 ConvertCGGITRBBinOp<cggi::OrOp, tfhe_rust_bool::OrOp>,
                 ConvertCGGITRBBinOp<cggi::NorOp, tfhe_rust_bool::NorOp>,
                 ConvertCGGITRBBinOp<cggi::XorOp, tfhe_rust_bool::XorOp>,
                 ConvertCGGITRBBinOp<cggi::XNorOp, tfhe_rust_bool::XnorOp>,
                 ConvertBoolEncodeOp, ConvertBoolTrivialEncryptOp,
                 ConvertBoolNotOp, ConvertPackedOp, ConvertAny<memref::AllocOp>,
                 ConvertAny<memref::DeallocOp>, ConvertAny<memref::StoreOp>,
                 ConvertAny<memref::LoadOp>, ConvertAny<memref::SubViewOp>,
                 ConvertAny<memref::CopyOp>, ConvertAny<tensor::FromElementsOp>,
                 ConvertAny<tensor::ExtractOp> >(typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
