#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h"

#include <utility>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "cggi-to-tfhe-rust"

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOTFHERUST
#include "lib/Dialect/CGGI/Conversions/CGGIToTfheRust/CGGIToTfheRust.h.inc"

constexpr int kBinaryGateLutWidth = 4;
constexpr int kAndLut = 8;
constexpr int kOrLut = 14;
constexpr int kXorLut = 6;

static Type encrytpedUIntTypeFromWidth(MLIRContext *ctx, int width) {
  // Only supporting unsigned types because the LWE dialect does not have a
  // notion of signedness.
  switch (width) {
    case 1:
      // The minimum bit width of the integer tfhe_rust API is UInt2
      // https://docs.rs/tfhe/latest/tfhe/index.html#types
      // This may happen if there are no LUT or boolean gate operations that
      // require a minimum bit width (e.g. shuffling bits in a program that
      // multiplies by two).
      LLVM_DEBUG(llvm::dbgs()
                 << "Upgrading ciphertext with bit width 1 to UInt2");
      [[fallthrough]];
    case 2:
      return tfhe_rust::EncryptedUInt2Type::get(ctx);
    case 3:
      return tfhe_rust::EncryptedUInt3Type::get(ctx);
    case 4:
      return tfhe_rust::EncryptedUInt4Type::get(ctx);
    case 8:
      return tfhe_rust::EncryptedUInt8Type::get(ctx);
    case 10:
      return tfhe_rust::EncryptedUInt10Type::get(ctx);
    case 12:
      return tfhe_rust::EncryptedUInt12Type::get(ctx);
    case 14:
      return tfhe_rust::EncryptedUInt14Type::get(ctx);
    case 16:
      return tfhe_rust::EncryptedUInt16Type::get(ctx);
    case 32:
      return tfhe_rust::EncryptedUInt32Type::get(ctx);
    case 64:
      return tfhe_rust::EncryptedUInt64Type::get(ctx);
    case 128:
      return tfhe_rust::EncryptedUInt128Type::get(ctx);
    case 256:
      return tfhe_rust::EncryptedUInt256Type::get(ctx);
    default:
      llvm_unreachable("Unsupported bitwidth");
  }
}

class CGGIToTfheRustTypeConverter : public TypeConverter {
 public:
  CGGIToTfheRustTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      int width = widthFromEncodingAttr(type.getEncoding());
      return encrytpedUIntTypeFromWidth(ctx, width);
    });
    addConversion([this](ShapedType type) -> Type {
      return type.cloneWith(type.getShape(),
                            this->convertType(type.getElementType()));
    });
  }
};

/// Returns the Value corresponding to a server key in the FuncOp containing
/// this op.
static FailureOr<Value> getContextualServerKey(Operation *op) {
  Value serverKey = op->getParentOfType<func::FuncOp>()
                        .getBody()
                        .getBlocks()
                        .front()
                        .getArguments()
                        .front();
  if (!mlir::isa<tfhe_rust::ServerKeyType>(serverKey.getType())) {
    return op->emitOpError()
           << "Found CGGI op in a function without a server "
              "key argument. Did the AddServerKeyArg pattern fail to run?";
  }
  return serverKey;
}

/// Convert a func by adding a server key argument. Converted ops in other
/// patterns need a server key SSA value available, so this pattern needs a
/// higher benefit.
struct AddServerKeyArg : public OpConversionPattern<func::FuncOp> {
  AddServerKeyArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsDialects<lwe::LWEDialect, cggi::CGGIDialect>(op)) {
      return failure();
    }

    auto serverKeyType = tfhe_rust::ServerKeyType::get(getContext());
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

struct AddServerKeyArgCall : public OpConversionPattern<func::CallOp> {
  AddServerKeyArgCall(mlir::MLIRContext *context)
      : OpConversionPattern<func::CallOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> sk = getContextualServerKey(op.getOperation());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    llvm::SmallVector<Value, 4> newOperands;
    newOperands.reserve(adaptor.getOperands().size() + 1);
    newOperands.push_back(sk.value());
    for (auto t : adaptor.getOperands()) {
      newOperands.push_back(t);
    }

    // // Set the updated operand list on the operation
    auto newCallOp = b.create<func::CallOp>(
        op.getLoc(), adaptor.getCallee(),
        getTypeConverter()->convertType(op.getResult(0).getType()),
        newOperands);
    rewriter.replaceOp(op, newCallOp);

    return success();
  }
};

/// Convert a Lut3Op to:
///   - generate_lookup_table
///   - scalar_left_shift
///   - add_op
///   - apply_lookup_table
///
/// Note the generated lookup tables are not uniqued across applications of this
/// pattern, so a separate step is required at the end to collect all the
/// identical lookup tables, and this can be done with a --cse pass.
struct ConvertLut3Op : public OpConversionPattern<cggi::Lut3Op> {
  ConvertLut3Op(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::Lut3Op>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::Lut3Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op.getOperation());
    if (failed(result)) return result;

    Value serverKey = result.value();
    // A followup -cse pass should combine repeated LUT generation ops.
    auto lut = b.create<tfhe_rust::GenerateLookupTableOp>(
        serverKey, adaptor.getLookupTable());
    // Construct input = c << 2 + b << 1 + a
    auto shiftedC = b.create<tfhe_rust::ScalarLeftShiftOp>(
        serverKey, adaptor.getC(),
        b.create<arith::ConstantOp>(b.getI8Type(), b.getI8IntegerAttr(2))
            .getResult());
    auto shiftedB = b.create<tfhe_rust::ScalarLeftShiftOp>(
        serverKey, adaptor.getB(),
        b.create<arith::ConstantOp>(b.getI8Type(), b.getI8IntegerAttr(1))
            .getResult());
    auto summedBC = b.create<tfhe_rust::AddOp>(serverKey, shiftedC, shiftedB);
    auto summedABC =
        b.create<tfhe_rust::AddOp>(serverKey, summedBC, adaptor.getA());

    rewriter.replaceOp(
        op, b.create<tfhe_rust::ApplyLookupTableOp>(serverKey, summedABC, lut));
    return success();
  }
};

struct ConvertLut2Op : public OpConversionPattern<cggi::Lut2Op> {
  ConvertLut2Op(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::Lut2Op>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::Lut2Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op.getOperation());
    if (failed(result)) return result;

    Value serverKey = result.value();
    // A followup -cse pass should combine repeated LUT generation ops.
    auto lut = b.create<tfhe_rust::GenerateLookupTableOp>(
        serverKey, adaptor.getLookupTable());
    // Construct input = b << 1 + a
    auto shiftedB = b.create<tfhe_rust::ScalarLeftShiftOp>(
        serverKey, adaptor.getB(),
        b.create<arith::ConstantOp>(b.getI8Type(), b.getI8IntegerAttr(1))
            .getResult());
    auto summedBA =
        b.create<tfhe_rust::AddOp>(serverKey, shiftedB, adaptor.getA());

    rewriter.replaceOp(
        op, b.create<tfhe_rust::ApplyLookupTableOp>(serverKey, summedBA, lut));
    return success();
  }
};

static LogicalResult replaceBinaryGate(Operation *op, Value lhs, Value rhs,
                                       ConversionPatternRewriter &rewriter,
                                       int lut) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  FailureOr<Value> result = getContextualServerKey(op);
  if (failed(result)) return result;

  Value serverKey = result.value();
  // A followup -cse pass should combine repeated LUT generation ops.
  auto lookupTable = b.getIntegerAttr(
      b.getIntegerType(kBinaryGateLutWidth, /*isSigned=*/false), lut);
  auto lutOp =
      b.create<tfhe_rust::GenerateLookupTableOp>(serverKey, lookupTable);
  // Construct input = rhs << 1 + lhs
  auto shiftedRhs = b.create<tfhe_rust::ScalarLeftShiftOp>(
      serverKey, rhs,
      b.create<arith::ConstantOp>(b.getI8Type(), b.getI8IntegerAttr(1))
          .getResult());
  auto input = b.create<tfhe_rust::AddOp>(serverKey, shiftedRhs, lhs);
  rewriter.replaceOp(
      op, b.create<tfhe_rust::ApplyLookupTableOp>(serverKey, input, lutOp));
  return success();
}

template <typename BinOp, typename TfheRustBinOp>
struct ConvertCGGITRBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(op, b.create<TfheRustBinOp>(serverKey, adaptor.getLhs(),
                                                   adaptor.getRhs()));
    return success();
  }
};

struct ConvertAndOp : public OpConversionPattern<cggi::AndOp> {
  ConvertAndOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::AndOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::AndOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return replaceBinaryGate(op.getOperation(), adaptor.getLhs(),
                             adaptor.getRhs(), rewriter, kAndLut);
  }
};

struct ConvertOrOp : public OpConversionPattern<cggi::OrOp> {
  ConvertOrOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::OrOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::OrOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return replaceBinaryGate(op.getOperation(), adaptor.getLhs(),
                             adaptor.getRhs(), rewriter, kOrLut);
  }
};

struct ConvertXorOp : public OpConversionPattern<cggi::XorOp> {
  ConvertXorOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::XorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::XorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return replaceBinaryGate(op.getOperation(), adaptor.getLhs(),
                             adaptor.getRhs(), rewriter, kXorLut);
  }
};

struct ConvertShROp : public OpConversionPattern<cggi::ShiftRightOp> {
  ConvertShROp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::ShiftRightOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::ShiftRightOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;
    Value serverKey = result.value();

    rewriter.replaceOpWithNewOp<tfhe_rust::ScalarRightShiftOp>(
        op, serverKey, adaptor.getLhs(), adaptor.getShiftAmount());

    return success();
  }
};

struct ConvertCastOp : public OpConversionPattern<cggi::CastOp> {
  ConvertCastOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::CastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;
    Value serverKey = result.value();

    auto outputType = getTypeConverter()->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<tfhe_rust::CastOp>(op, outputType, serverKey,
                                                   adaptor.getInput());

    return success();
  }
};

struct ConvertNotOp : public OpConversionPattern<cggi::NotOp> {
  ConvertNotOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::NotOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::NotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;
    Value serverKey = result.value();

    auto shapedTy = dyn_cast<ShapedType>(op.getInput().getType());
    Type eltTy = shapedTy ? shapedTy.getElementType() : op.getInput().getType();

    auto width = widthFromEncodingAttr(
        cast<lwe::LWECiphertextType>(eltTy).getEncoding());
    auto cleartextType = b.getIntegerType(width);
    auto outputType = encrytpedUIntTypeFromWidth(getContext(), width);

    // not(x) == trivial_encryption(1) - x
    Value createTrivialOp = b.create<tfhe_rust::CreateTrivialOp>(
        outputType, serverKey,
        b.create<arith::ConstantOp>(cleartextType,
                                    b.getIntegerAttr(cleartextType, 1))
            .getResult());
    if (shapedTy) {
      createTrivialOp = b.create<tensor::FromElementsOp>(
          shapedTy,
          SmallVector<Value>(shapedTy.getNumElements(), createTrivialOp));
    }
    rewriter.replaceOp(op, b.create<tfhe_rust::SubOp>(
                               serverKey, createTrivialOp, adaptor.getInput()));
    return success();
  }
};

struct ConvertTrivialEncryptOp
    : public OpConversionPattern<lwe::TrivialEncryptOp> {
  ConvertTrivialEncryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::TrivialEncryptOp>(context, /*benefit=*/2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::TrivialEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualServerKey(op.getOperation());
    if (failed(result)) return result;

    Value serverKey = result.value();
    lwe::EncodeOp encodeOp = op.getInput().getDefiningOp<lwe::EncodeOp>();
    if (!encodeOp) {
      return op.emitError() << "Expected input to TrivialEncrypt to be the "
                               "result of an EncodeOp, but it was "
                            << op.getInput().getDefiningOp()->getName();
    }
    auto outputType = encrytpedUIntTypeFromWidth(
        getContext(), widthFromEncodingAttr(encodeOp.getEncoding()));
    auto createTrivialOp = rewriter.create<tfhe_rust::CreateTrivialOp>(
        op.getLoc(), outputType, serverKey, encodeOp.getInput());
    rewriter.replaceOp(op, createTrivialOp);
    return success();
  }
};

struct ConvertTrivialOp : public OpConversionPattern<cggi::CreateTrivialOp> {
  ConvertTrivialOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::CreateTrivialOp>(context, /*benefit=*/2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::CreateTrivialOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualServerKey(op.getOperation());
    if (failed(result)) return result;

    Value serverKey = result.value();

    auto intValue = op.getValue().getValue().getSExtValue();
    auto inputValue = mlir::IntegerAttr::get(op.getValue().getType(), intValue);
    auto constantWidth = op.getValue().getValue().getBitWidth();

    auto cteOp = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerType(constantWidth), inputValue);

    auto outputType = encrytpedUIntTypeFromWidth(getContext(), constantWidth);

    auto createTrivialOp = rewriter.create<tfhe_rust::CreateTrivialOp>(
        op.getLoc(), outputType, serverKey, cteOp);
    rewriter.replaceOp(op, createTrivialOp);
    return success();
  }
};

struct ConvertEncodeOp : public OpConversionPattern<lwe::EncodeOp> {
  ConvertEncodeOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::EncodeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::EncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class CGGIToTfheRust : public impl::CGGIToTfheRustBase<CGGIToTfheRust> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *op = getOperation();

    CGGIToTfheRustTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addLegalDialect<tfhe_rust::TfheRustDialect>();
    target.addIllegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();

    // FuncOp is marked legal by the default structural conversion patterns
    // helper, just based on type conversion. We need more, but because the
    // addDynamicallyLegalOp is a set-based method, we can add this after
    // calling addStructuralConversionPatterns and it will overwrite the
    // legality condition set in that function.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasServerKeyArg = op.getFunctionType().getNumInputs() > 0 &&
                             mlir::isa<tfhe_rust::ServerKeyType>(
                                 *op.getFunctionType().getInputs().begin());
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsDialects<lwe::LWEDialect, cggi::CGGIDialect>(op) ||
              hasServerKeyArg);
    });

    target.addLegalOp<mlir::arith::ConstantOp>();

    target.addDynamicallyLegalOp<
        memref::AllocOp, memref::DeallocOp, memref::StoreOp, memref::LoadOp,
        memref::SubViewOp, memref::CopyOp, affine::AffineLoadOp,
        affine::AffineStoreOp, tensor::FromElementsOp, tensor::ExtractOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });

    // FIXME: still need to update callers to insert the new server key arg, if
    // needed and possible.
    patterns.add<
        AddServerKeyArg, ConvertEncodeOp, ConvertLut2Op, ConvertLut3Op,
        ConvertNotOp, ConvertTrivialEncryptOp, ConvertTrivialOp,
        ConvertCGGITRBinOp<cggi::AddOp, tfhe_rust::AddOp>,
        ConvertCGGITRBinOp<cggi::MulOp, tfhe_rust::MulOp>,
        ConvertCGGITRBinOp<cggi::SubOp, tfhe_rust::SubOp>, ConvertAndOp,
        ConvertOrOp, ConvertXorOp, ConvertCastOp, ConvertShROp,
        ConvertAny<memref::AllocOp>, ConvertAny<memref::DeallocOp>,
        ConvertAny<memref::StoreOp>, ConvertAny<memref::LoadOp>,
        ConvertAny<memref::SubViewOp>, ConvertAny<memref::CopyOp>,
        ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::ExtractOp>,
        ConvertAny<affine::AffineLoadOp>, ConvertAny<affine::AffineStoreOp>>(
        typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
