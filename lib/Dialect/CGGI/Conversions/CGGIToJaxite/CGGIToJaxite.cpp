#include "lib/Dialect/CGGI/Conversions/CGGIToJaxite/CGGIToJaxite.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteOps.h"
#include "lib/Dialect/Jaxite/IR/JaxiteTypes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOJAXITE
#include "lib/Dialect/CGGI/Conversions/CGGIToJaxite/CGGIToJaxite.h.inc"

class CGGIToJaxiteTypeConverter : public TypeConverter {
 public:
  CGGIToJaxiteTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](lwe::LWECiphertextType type) -> Type {
      if (type.getPlaintextSpace()
              .getRing()
              .getCoefficientType()
              .getIntOrFloatBitWidth() == 3) {
        return type;
      }
      llvm_unreachable("Unsupported cleartext bitwidth in jaxite");
      return 0;
    });
    addConversion([this](ShapedType type) -> Type {
      return type.cloneWith(type.getShape(),
                            this->convertType(type.getElementType()));
    });
  }
};

// Returns the Value corresponding to a JaxiteArgType in the FuncOp containing
// this op.
template <typename JaxiteArgType>
FailureOr<Value> getContextualJaxiteArg(Operation* op) {
  auto result = getContextualArgFromFunc<JaxiteArgType>(op);
  if (failed(result)) {
    return op->emitOpError() << "Cannot find Jaxite server argument. Did the "
                                "AddJaxiteContextualArgs pattern fail to run?";
  }
  return result.value();
}

/// Convert a func by adding contextual server args. Converted ops in other
/// patterns need a server key and params SSA values available, so this pattern
/// needs a higher benefit.
struct AddJaxiteContextualArgs : public OpConversionPattern<func::FuncOp> {
  AddJaxiteContextualArgs(mlir::MLIRContext* context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 5) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!containsDialects<lwe::LWEDialect, cggi::CGGIDialect>(op)) {
      return rewriter.notifyMatchFailure(
          op, "op does not contain lwe or cggi dialects");
    }

    auto serverKeyType = jaxite::ServerKeySetType::get(getContext());
    auto paramsType = jaxite::ParamsType::get(getContext());

    // Insert all argument at the ending
    // NOTE: arguments with identical index will
    // appear in the same order that they were listed.
    SmallVector<unsigned> argIndices(2, op.getNumArguments());
    SmallVector<DictionaryAttr> argAttrs(2, nullptr);
    SmallVector<Location> argLocs(2, op.getLoc());
    rewriter.modifyOpInPlace(op, [&] {
      (void)op.insertArguments(argIndices, {serverKeyType, paramsType},
                               argAttrs, argLocs);
    });
    return success();
  }
};

struct ConvertCGGIToJaxiteLut3Op : public OpConversionPattern<cggi::Lut3Op> {
  ConvertCGGIToJaxiteLut3Op(mlir::MLIRContext* context)
      : OpConversionPattern<cggi::Lut3Op>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::Lut3Op op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> resultServerKey =
        getContextualJaxiteArg<jaxite::ServerKeySetType>(op.getOperation());
    if (failed(resultServerKey)) return resultServerKey;
    Value serverKey = resultServerKey.value();

    FailureOr<Value> resultParams =
        getContextualJaxiteArg<jaxite::ParamsType>(op.getOperation());
    if (failed(resultParams)) return resultParams;
    Value params = resultParams.value();

    int64_t truthTableValue =
        static_cast<uint8_t>(op.getLookupTableAttr().getUInt());
    Value tt = arith::ConstantOp::create(
        b, op.getLoc(), b.getIntegerAttr(b.getI8Type(), truthTableValue));

    // The ciphertext parameters (a, b, c) are passed in reverse order from cggi
    // to jaxite to mirror jaxite API
    auto createLut3Op = jaxite::Lut3Op::create(
        rewriter, op.getLoc(),
        typeConverter->convertType(op.getOutput().getType()), adaptor.getA(),
        adaptor.getB(), adaptor.getC(), tt, serverKey, params);
    rewriter.replaceOp(op, createLut3Op);
    return success();
  }
};

struct ConvertCGGIToJaxitePmapLut3Op
    : public OpConversionPattern<cggi::PackedLut3Op> {
  ConvertCGGIToJaxitePmapLut3Op(mlir::MLIRContext* context)
      : OpConversionPattern<cggi::PackedLut3Op>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::PackedLut3Op op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    FailureOr<Value> resultServerKey =
        getContextualJaxiteArg<jaxite::ServerKeySetType>(op.getOperation());
    if (failed(resultServerKey)) return resultServerKey;
    Value serverKey = resultServerKey.value();

    FailureOr<Value> resultParams =
        getContextualJaxiteArg<jaxite::ParamsType>(op.getOperation());
    if (failed(resultParams)) return resultParams;
    Value params = resultParams.value();
    auto A = op.getA();
    auto B = op.getB();
    auto C = op.getC();
    auto truthTableValues = op.getLookupTables();
    SmallVector<Value> lut3_args;
    for (int i = 0; i < truthTableValues.size(); ++i) {
      uint8_t truthTableValue = static_cast<uint8_t>(
          cast<IntegerAttr>(truthTableValues[i]).getUInt());
      auto tt = arith::ConstantOp::create(
          b, op.getLoc(), b.getIntegerAttr(b.getI8Type(), truthTableValue));
      auto extractionIndex =
          arith::ConstantOp::create(b, op->getLoc(), b.getIndexAttr(i));
      auto extractOpA = tensor::ExtractOp::create(b, op->getLoc(), A,
                                                  extractionIndex.getResult());
      auto extractOpB = tensor::ExtractOp::create(b, op->getLoc(), B,
                                                  extractionIndex.getResult());
      auto extractOpC = tensor::ExtractOp::create(b, op->getLoc(), C,
                                                  extractionIndex.getResult());
      auto lut3ArgsOp = jaxite::Lut3ArgsOp::create(b, op.getLoc(), extractOpA,
                                                   extractOpB, extractOpC, tt);
      lut3_args.push_back(lut3ArgsOp.getResult());
    }
    auto lut3ArgsOp =
        tensor::FromElementsOp::create(b, op->getLoc(), lut3_args);
    rewriter.replaceOpWithNewOp<jaxite::PmapLut3Op>(
        op, op.getOutput().getType(), lut3ArgsOp.getResult(), serverKey,
        params);
    return success();
  }
};

struct ConvertCGGIToJaxiteTrivialEncryptOp
    : public OpConversionPattern<lwe::TrivialEncryptOp> {
  ConvertCGGIToJaxiteTrivialEncryptOp(mlir::MLIRContext* context)
      : OpConversionPattern<lwe::TrivialEncryptOp>(context, /*benefit=*/10) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::TrivialEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualJaxiteArg<jaxite::ParamsType>(op.getOperation());
    if (failed(result)) return result;
    Value serverParams = result.value();

    // Find the EncodeOp that should be feeding this TrivialEncrypt
    lwe::EncodeOp encodeOp = op.getInput().getDefiningOp<lwe::EncodeOp>();
    if (!encodeOp || !encodeOp.getInput().getType().isSignlessIntOrFloat() ||
        encodeOp.getInput().getType().getIntOrFloatBitWidth() != 1) {
      return op.emitError() << "Expected input to TrivialEncrypt to be a "
                               "boolean LWEPlaintext but found "
                            << op.getInput().getType();
    }

    auto createConstantOp = jaxite::ConstantOp::create(
        rewriter, op.getLoc(),
        typeConverter->convertType(op.getOutput().getType()),
        encodeOp.getInput(), serverParams);
    rewriter.replaceOp(op, createConstantOp);

    // Erase the EncodeOp if it has no other users
    if (encodeOp.getResult().hasOneUse()) {
      rewriter.eraseOp(encodeOp);
    }
    return success();
  }
};

class CGGIToJaxite : public impl::CGGIToJaxiteBase<CGGIToJaxite> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* op = getOperation();

    CGGIToJaxiteTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addLegalDialect<jaxite::JaxiteDialect>();
    target.addIllegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();
    // Mark EncodeOp as legal - we'll handle it within TrivialEncrypt patterns
    target.addLegalOp<lwe::EncodeOp>();
    // FuncOp is marked legal by the default structural conversion patterns
    // helper, just based on type conversion. We need more, but because the
    // addDynamicallyLegalOp is a set-based method, we can add this after
    // calling addStructuralConversionPatterns and it will overwrite the
    // legality condition set in that function.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      int num_inputs = op.getFunctionType().getNumInputs();
      if (num_inputs < 2) {
        return false;
      }
      bool hasServerKeyArg = llvm::any_of(op.getArgumentTypes(), [](Type t) {
        return isa<jaxite::ServerKeySetType>(t);
      });
      bool hasParamsArg = llvm::any_of(op.getArgumentTypes(), [](Type t) {
        return isa<jaxite::ParamsType>(t);
      });
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) && hasServerKeyArg &&
             hasParamsArg;
    });

    patterns.add<AddJaxiteContextualArgs, ConvertCGGIToJaxiteLut3Op,
                 ConvertCGGIToJaxiteTrivialEncryptOp,
                 ConvertCGGIToJaxitePmapLut3Op, ConvertAny<>>(typeConverter,
                                                              context);
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(
            applyPartialConversion(op, target, std::move(patterns), config))) {
      return signalPassFailure();
    }

    // Post-conversion cleanup: remove any remaining EncodeOps
    // These would be EncodeOps that are not used by any TrivialEncrypt
    // operations
    SmallVector<lwe::EncodeOp> encodeOpsToErase;
    op->walk([&](lwe::EncodeOp encodeOp) {
      if (encodeOp.use_empty()) {
        encodeOpsToErase.push_back(encodeOp);
      } else {
        // This shouldn't happen - all encode ops should have been
        // erased by ConvertBoolTrivialEncryptOp
        encodeOp.emitError()
            << "EncodeOp with TrivialEncrypt users found after conversion - "
               "this indicates a TrivialEncrypt pattern failed to run";
        signalPassFailure();
      }
    });

    // Erase the unused EncodeOps
    for (auto encodeOp : encodeOpsToErase) {
      encodeOp.erase();
    }
  }
};

}  // namespace mlir::heir
