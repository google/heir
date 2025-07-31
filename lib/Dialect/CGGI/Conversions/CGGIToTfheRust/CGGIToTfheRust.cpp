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
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
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

class CGGIToTfheRustTypeConverter : public TypeConverter {
 public:
  CGGIToTfheRustTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      int width = type.getPlaintextSpace()
                      .getRing()
                      .getCoefficientType()
                      .getIntOrFloatBitWidth();
      return encrytpedUIntTypeFromWidth(ctx, width);
    });
    addConversion([this](ShapedType type) -> Type {
      auto elemType = this->convertType(type.getElementType());
      if (auto rankedTensorTy = dyn_cast<RankedTensorType>(type))
        return RankedTensorType::get(rankedTensorTy.getShape(), elemType);
      if (auto memrefTy = dyn_cast<MemRefType>(type))
        return MemRefType::get(memrefTy.getShape(), elemType);
      return type.cloneWith(type.getShape(), elemType);
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

    rewriter.modifyOpInPlace(op, [&] {
      (void)op.insertArgument(0, serverKeyType, nullptr, op.getLoc());
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
    auto newCallOp = func::CallOp::create(
        b, op.getLoc(), adaptor.getCallee(),
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
    auto lut = tfhe_rust::GenerateLookupTableOp::create(
        b, serverKey, adaptor.getLookupTable());
    // Construct input = c << 2 + b << 1 + a
    auto shiftedC = tfhe_rust::ScalarLeftShiftOp::create(
        b, serverKey, adaptor.getC(), b.getIndexAttr(2));
    auto shiftedB = tfhe_rust::ScalarLeftShiftOp::create(
        b, serverKey, adaptor.getB(), b.getIndexAttr(1));
    auto summedBC = tfhe_rust::AddOp::create(b, adaptor.getB().getType(),
                                             serverKey, shiftedC, shiftedB);
    auto summedABC = tfhe_rust::AddOp::create(
        b, adaptor.getB().getType(), serverKey, summedBC, adaptor.getA());

    rewriter.replaceOp(op, tfhe_rust::ApplyLookupTableOp::create(
                               b, serverKey, summedABC, lut));
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
    auto lut = tfhe_rust::GenerateLookupTableOp::create(
        b, serverKey, adaptor.getLookupTable());
    // Construct input = b << 1 + a
    auto shiftedB = tfhe_rust::ScalarLeftShiftOp::create(
        b, serverKey, adaptor.getB(), b.getIndexAttr(1));

    auto summedBA = tfhe_rust::AddOp::create(
        b, getTypeConverter()->convertType(shiftedB.getResult().getType()),
        serverKey, shiftedB, adaptor.getA());

    rewriter.replaceOp(
        op, tfhe_rust::ApplyLookupTableOp::create(b, serverKey, summedBA, lut));
    return success();
  }
};

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
    CGGIToTfheRustTypeConverter typeConverter(op->getContext());
    auto outputType = typeConverter.convertType(op.getResult().getType());

    rewriter.replaceOp(
        op, TfheRustBinOp::create(b, outputType, serverKey, adaptor.getLhs(),
                                  adaptor.getRhs()));
    return success();
  }
};

struct ConvertScalarMulOp : public OpConversionPattern<lwe::MulScalarOp> {
  using OpConversionPattern<lwe::MulScalarOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::MulScalarOp op, lwe::MulScalarOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();
    CGGIToTfheRustTypeConverter typeConverter(op->getContext());
    auto outputType = typeConverter.convertType(op.getResult().getType());

    rewriter.replaceOp(op, tfhe_rust::MulOp::create(b, outputType, serverKey,
                                                    adaptor.getCiphertext(),
                                                    adaptor.getScalar()));
    return success();
  }
};

template <typename BinOp, typename TfheRustBinOp>
struct ConvertCGGICtxtBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();
    CGGIToTfheRustTypeConverter typeConverter(op->getContext());
    auto outputType = typeConverter.convertType(op.getResult().getType());

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (lhs.getType() != rhs.getType()) {
      if (!isa<lwe::LWECiphertextType>(op.getLhs().getType())) {
        lhs = tfhe_rust::CreateTrivialOp::create(b, outputType, serverKey, lhs);
      } else if (!isa<lwe::LWECiphertextType>(op.getRhs().getType())) {
        rhs = tfhe_rust::CreateTrivialOp::create(b, outputType, serverKey, rhs);
      } else {
        return op.emitError()
               << "Expected both operands to be of the same type";
      }
    }

    rewriter.replaceOp(
        op, TfheRustBinOp::create(b, outputType, serverKey, lhs, rhs));
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<cggi::SelectOp> {
  using OpConversionPattern<cggi::SelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::SelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(op, tfhe_rust::SelectOp::create(
                               b, adaptor.getTrueCtxt().getType(), serverKey,
                               adaptor.getSelect(), adaptor.getTrueCtxt(),
                               adaptor.getFalseCtxt()));
    return success();
  }
};

struct ConvertCmpOp : public OpConversionPattern<cggi::CmpOp> {
  using OpConversionPattern<cggi::CmpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::CmpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;

    Value serverKey = result.value();

    rewriter.replaceOp(
        op,
        tfhe_rust::CmpOp::create(
            b, tfhe_rust::EncryptedBoolType::get(op->getContext()), serverKey,
            adaptor.getPredicate(), adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ConvertShROp : public OpConversionPattern<cggi::ScalarShiftRightOp> {
  ConvertShROp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::ScalarShiftRightOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::ScalarShiftRightOp op, OpAdaptor adaptor,
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

struct ConvertShLOp : public OpConversionPattern<cggi::ScalarShiftLeftOp> {
  ConvertShLOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::ScalarShiftLeftOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::ScalarShiftLeftOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> result = getContextualServerKey(op);
    if (failed(result)) return result;
    Value serverKey = result.value();

    rewriter.replaceOpWithNewOp<tfhe_rust::ScalarLeftShiftOp>(
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

    auto width = cast<lwe::LWECiphertextType>(eltTy)
                     .getPlaintextSpace()
                     .getRing()
                     .getCoefficientType()
                     .getIntOrFloatBitWidth();
    auto cleartextType = b.getIntegerType(width);
    auto outputType = encrytpedUIntTypeFromWidth(getContext(), width);

    // not(x) == trivial_encryption(1) - x
    Value createTrivialOp = tfhe_rust::CreateTrivialOp::create(
        b, outputType, serverKey,
        arith::ConstantOp::create(b, cleartextType,
                                  b.getIntegerAttr(cleartextType, 1))
            .getResult());
    if (shapedTy) {
      createTrivialOp = tensor::FromElementsOp::create(
          b, shapedTy,
          SmallVector<Value>(shapedTy.getNumElements(), createTrivialOp));
    }
    rewriter.replaceOp(op,
                       tfhe_rust::SubOp::create(b, serverKey, createTrivialOp,
                                                adaptor.getInput()));
    return success();
  }
};

struct ConvertProgrammableBootstrapOp
    : public OpConversionPattern<cggi::ProgrammableBootstrapOp> {
  ConvertProgrammableBootstrapOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::ProgrammableBootstrapOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::ProgrammableBootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualServerKey(op.getOperation());
    if (failed(result)) return result;
    Value serverKey = result.value();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto lutOp = tfhe_rust::GenerateLookupTableOp::create(b, serverKey,
                                                          op.getLookupTable());
    rewriter.replaceOp(op, tfhe_rust::ApplyLookupTableOp::create(
                               b, serverKey, adaptor.getInput(), lutOp));
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
        getContext(), op.getInput()
                          .getDefiningOp<lwe::EncodeOp>()
                          .getOutput()
                          .getType()
                          .getPlaintextSpace()
                          .getRing()
                          .getCoefficientType()
                          .getIntOrFloatBitWidth());
    auto createTrivialOp = tfhe_rust::CreateTrivialOp::create(
        rewriter, op.getLoc(), outputType, serverKey, encodeOp.getInput());
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

    auto cteOp = arith::ConstantOp::create(rewriter, op.getLoc(),
                                           op.getValue().getType(), inputValue);

    auto outputType = encrytpedUIntTypeFromWidth(getContext(), constantWidth);

    if (auto rankedTensorTy =
            dyn_cast<RankedTensorType>(op.getResult().getType())) {
      auto shape = rankedTensorTy.getShape();
      outputType = RankedTensorType::get(shape, outputType);
    }

    auto createTrivialOp = tfhe_rust::CreateTrivialOp::create(
        rewriter, op.getLoc(), outputType, serverKey, cteOp);
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

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      bool hasServerKeyArg =
          isa<tfhe_rust::ServerKeyType>(op.getOperand(0).getType());
      return hasServerKeyArg;
    });

    target.addLegalOp<mlir::arith::ConstantOp>();

    target.addDynamicallyLegalOp<
        memref::AllocOp, memref::DeallocOp, memref::StoreOp, memref::LoadOp,
        memref::SubViewOp, memref::CopyOp, affine::AffineLoadOp,
        tensor::InsertOp, tensor::InsertSliceOp, affine::AffineStoreOp,
        tensor::FromElementsOp, tensor::ExtractOp>([&](Operation *op) {
      return typeConverter.isLegal(op->getOperandTypes()) &&
             typeConverter.isLegal(op->getResultTypes());
    });

    patterns.add<
        AddServerKeyArg, AddServerKeyArgCall, ConvertEncodeOp, ConvertLut2Op,
        ConvertLut3Op, ConvertNotOp, ConvertTrivialEncryptOp, ConvertTrivialOp,
        ConvertCGGITRBinOp<lwe::AddOp, tfhe_rust::AddOp>, ConvertScalarMulOp,
        ConvertCGGITRBinOp<cggi::AddOp, tfhe_rust::AddOp>,
        ConvertCGGITRBinOp<cggi::MulOp, tfhe_rust::MulOp>,
        ConvertCGGITRBinOp<cggi::SubOp, tfhe_rust::SubOp>,
        ConvertCGGITRBinOp<cggi::SubOp, tfhe_rust::SubOp>,
        ConvertCGGICtxtBinOp<cggi::EqOp, tfhe_rust::EqOp>,
        ConvertCGGICtxtBinOp<cggi::NeqOp, tfhe_rust::NeqOp>,
        ConvertCGGICtxtBinOp<cggi::MinOp, tfhe_rust::MinOp>,
        ConvertCGGICtxtBinOp<cggi::MaxOp, tfhe_rust::MaxOp>, ConvertSelectOp,
        ConvertCmpOp, ConvertCGGICtxtBinOp<cggi::AndOp, tfhe_rust::BitAndOp>,
        ConvertCGGICtxtBinOp<cggi::OrOp, tfhe_rust::BitOrOp>,
        ConvertCGGICtxtBinOp<cggi::XorOp, tfhe_rust::BitXorOp>, ConvertCastOp,
        ConvertShROp, ConvertShLOp, ConvertAny<memref::AllocOp>,
        ConvertProgrammableBootstrapOp, ConvertAny<memref::DeallocOp>,
        ConvertAny<memref::StoreOp>, ConvertAny<memref::LoadOp>,
        ConvertAny<memref::SubViewOp>, ConvertAny<memref::CopyOp>,
        ConvertAny<tensor::InsertOp>, ConvertAny<tensor::InsertSliceOp>,
        ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::ExtractOp>,
        ConvertAny<affine::AffineLoadOp>, ConvertAny<affine::AffineStoreOp>>(
        typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
