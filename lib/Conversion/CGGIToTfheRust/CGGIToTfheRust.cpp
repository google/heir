#include "lib/Conversion/CGGIToTfheRust/CGGIToTfheRust.h"

#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
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

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOTFHERUST
#include "lib/Conversion/CGGIToTfheRust/CGGIToTfheRust.h.inc"

constexpr int kBinaryGateLutWidth = 4;
constexpr int kAndLut = 8;
constexpr int kOrLut = 14;
constexpr int kXorLut = 6;

Type encrytpedUIntTypeFromWidth(MLIRContext *ctx, int width) {
  // Only supporting unsigned types because the LWE dialect does not have a
  // notion of signedness.
  switch (width) {
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

/// Returns true if the func's body contains any CGGI ops.
bool containsCGGIOps(func::FuncOp func) {
  auto walkResult = func.walk([&](Operation *op) {
    if (llvm::isa<cggi::CGGIDialect>(op->getDialect()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

/// Returns the Value corresponding to a server key in the FuncOp containing
/// this op.
FailureOr<Value> getContextualServerKey(Operation *op) {
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
struct AddServerKeyArg : public OpConversionPattern<func::FuncOp> {
  AddServerKeyArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsCGGIOps(op)) {
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

LogicalResult replaceBinaryGate(Operation *op, Value lhs, Value rhs,
                                ConversionPatternRewriter &rewriter, int lut) {
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
        op.getLoc(), outputType, serverKey, encodeOp.getPlaintext());
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
             (!containsCGGIOps(op) || hasServerKeyArg);
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
        AddServerKeyArg, ConvertAndOp, ConvertEncodeOp, ConvertLut2Op,
        ConvertLut3Op, ConvertNotOp, ConvertOrOp, ConvertTrivialEncryptOp,
        ConvertXorOp, GenericOpPattern<memref::AllocOp>,
        GenericOpPattern<memref::DeallocOp>, GenericOpPattern<memref::StoreOp>,
        GenericOpPattern<memref::LoadOp>, GenericOpPattern<memref::SubViewOp>,
        GenericOpPattern<memref::CopyOp>,
        GenericOpPattern<tensor::FromElementsOp>,
        GenericOpPattern<tensor::ExtractOp>>(typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
