#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOOPENFHE
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h.inc"

ToOpenfheTypeConverter::ToOpenfheTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  addConversion([ctx](lwe::RLWEPublicKeyType type) -> Type {
    return openfhe::PublicKeyType::get(ctx);
  });
  addConversion([ctx](lwe::RLWESecretKeyType type) -> Type {
    return openfhe::PrivateKeyType::get(ctx);
  });
  addConversion([ctx](lwe::NewLWEPublicKeyType type) -> Type {
    return openfhe::PublicKeyType::get(ctx);
  });
  addConversion([ctx](lwe::NewLWESecretKeyType type) -> Type {
    return openfhe::PrivateKeyType::get(ctx);
  });
}

FailureOr<Value> getContextualCryptoContext(Operation *op) {
  auto result = getContextualArgFromFunc<openfhe::CryptoContextType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found LWE op in a function without a public key argument."
              " Did the AddCryptoContextArg pattern fail to run?";
  }
  return result.value();
}

// NOTE: we can not use containsDialect
// for FuncOp declaration, which does not have a body
template <typename... Dialects>
bool containsArgumentOfDialect(func::FuncOp funcOp) {
  return llvm::any_of(funcOp.getArgumentTypes(), [&](Type argType) {
    return DialectEqual<Dialects...>()(&argType.getDialect());
  });
}

struct AddCryptoContextArg : public OpConversionPattern<func::FuncOp> {
  AddCryptoContextArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto containsCryptoOps =
        containsDialects<lwe::LWEDialect, bgv::BGVDialect, ckks::CKKSDialect>(
            op);
    auto containsCryptoArg =
        containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect,
                                  ckks::CKKSDialect>(op);
    if (!(containsCryptoOps || containsCryptoArg)) {
      return failure();
    }

    auto cryptoContextType = openfhe::CryptoContextType::get(getContext());
    FunctionType originalType = op.getFunctionType();
    llvm::SmallVector<Type, 4> newTypes;
    newTypes.reserve(originalType.getNumInputs() + 1);
    newTypes.push_back(cryptoContextType);
    for (auto t : originalType.getInputs()) {
      newTypes.push_back(t);
    }
    auto newFuncType =
        FunctionType::get(getContext(), newTypes, originalType.getResults());
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(newFuncType);

      // guard against FuncOp declaration
      if (!op.isDeclaration()) {
        Block &block = op.getBody().getBlocks().front();
        block.insertArgument(&block.getArguments().front(), cryptoContextType,
                             op.getLoc());
      }
    });

    return success();
  }
};

struct ConvertFuncCallOp : public OpConversionPattern<func::CallOp> {
  ConvertFuncCallOp(mlir::MLIRContext *context)
      : OpConversionPattern<func::CallOp>(context) {}

  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;
    auto cryptoContext = result.value();

    auto callee = op.getCallee();
    auto operands = adaptor.getOperands();
    auto resultTypes = op.getResultTypes();

    SmallVector<Value> newOperands;
    newOperands.push_back(cryptoContext);
    for (auto operand : operands) {
      newOperands.push_back(operand);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes,
                                              newOperands);
    return success();
  }
};

struct ConvertEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  ConvertEncryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::RLWEEncryptOp>(context) {}

  using OpConversionPattern<lwe::RLWEEncryptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, typename lwe::RLWEEncryptOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    auto keyType = dyn_cast<lwe::NewLWEPublicKeyType>(op.getKey().getType());
    if (!keyType)
      return op.emitError()
             << "OpenFHE only supports public key encryption for LWE.";

    Value cryptoContext = result.value();
    rewriter.replaceOp(op,
                       rewriter.create<openfhe::EncryptOp>(
                           op.getLoc(), op.getOutput().getType(), cryptoContext,
                           adaptor.getInput(), adaptor.getKey()));
    return success();
  }
};

struct ConvertDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  ConvertDecryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<RLWEDecryptOp>(context) {}

  using OpConversionPattern<RLWEDecryptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RLWEDecryptOp op, RLWEDecryptOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op,
                       rewriter.create<openfhe::DecryptOp>(
                           op.getLoc(), op.getOutput().getType(), cryptoContext,
                           adaptor.getInput(), adaptor.getSecretKey()));
    return success();
  }
};

struct ConvertEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  explicit ConvertEncodeOp(const mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context)
      : mlir::OpConversionPattern<lwe::RLWEEncodeOp>(typeConverter, context) {}

  // OpenFHE has a convention that all inputs to MakePackedPlaintext are
  // std::vector<int64_t>, so we need to cast the input to that type.
  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;
    Value cryptoContext = result.value();

    Value input = adaptor.getInput();
    auto elementTy = getElementTypeOrSelf(input.getType());

    auto tensorTy = mlir::dyn_cast<RankedTensorType>(input.getType());
    // Replicate scalar inputs into a splat tensor with shape matching
    // the ring dimension.
    if (!tensorTy) {
      auto ringDegree =
          op.getRing().getPolynomialModulus().getPolynomial().getDegree();
      tensor::SplatOp splat = rewriter.create<tensor::SplatOp>(
          op.getLoc(), RankedTensorType::get({ringDegree}, elementTy), input);
      input = splat.getResult();
      tensorTy = splat.getType();
    }

    // Cast inputs to the correct types for OpenFHE API.
    if (auto intTy = mlir::dyn_cast<IntegerType>(elementTy)) {
      if (intTy.getWidth() > 64)
        return op.emitError() << "No supported packing technique for integers "
                                 "bigger than 64 bits.";

      if (intTy.getWidth() < 64) {
        // OpenFHE has a convention that all inputs to MakePackedPlaintext are
        // std::vector<int64_t>, so we need to cast the input to that type.
        auto int64Ty = rewriter.getIntegerType(64);
        auto newTensorTy = RankedTensorType::get(tensorTy.getShape(), int64Ty);
        input =
            rewriter.create<arith::ExtSIOp>(op.getLoc(), newTensorTy, input);
      }
    } else {
      auto floatTy = cast<FloatType>(elementTy);
      if (floatTy.getWidth() > 64)
        return op.emitError() << "No supported packing technique for floats "
                                 "bigger than 64 bits.";

      if (floatTy.getWidth() < 64) {
        // OpenFHE has a convention that all inputs to MakeCKKSPackedPlaintext
        // are std::vector<double>, so we need to cast the input to that type.
        auto f64Ty = rewriter.getF64Type();
        auto newTensorTy = RankedTensorType::get(tensorTy.getShape(), f64Ty);
        input = rewriter.create<arith::ExtFOp>(op.getLoc(), newTensorTy, input);
      }
    }

    lwe::NewLWEPlaintextType plaintextType = lwe::NewLWEPlaintextType::get(
        op.getContext(),
        lwe::ApplicationDataAttr::get(adaptor.getInput().getType(),
                                      lwe::NoOverflowAttr::get(getContext())),
        lwe::PlaintextSpaceAttr::get(getContext(), op.getRing(),
                                     op.getEncoding()));

    return llvm::TypeSwitch<Attribute, LogicalResult>(op.getEncoding())
        .Case<lwe::InverseCanonicalEncodingAttr>([&](auto encoding) {
          rewriter.replaceOpWithNewOp<openfhe::MakeCKKSPackedPlaintextOp>(
              op, plaintextType, cryptoContext, input);
          return success();
        })
        .Case<lwe::CoefficientEncodingAttr>([&](auto encoding) {
          // TODO (#1192): support coefficient packing in `--lwe-to-openfhe`
          op.emitError() << "HEIR does not yet support coefficient encoding "
                            " when targeting OpenFHE";
          return failure();
        })
        .Case<lwe::FullCRTPackingEncodingAttr>([&](auto encoding) {
          rewriter.replaceOpWithNewOp<openfhe::MakePackedPlaintextOp>(
              op, plaintextType, cryptoContext, input);
          return success();
        })
        .Default([&](Attribute) -> LogicalResult {
          // encoding isn't support explicitly:
          op.emitError(
              "Unexpected encoding while targeting OpenFHE. "
              "If you expect this type of encoding to be supported "
              "for the OpenFHE backend, please file a bug report.");
          return failure();
        });
  }
};

struct ConvertBootstrapOp : public OpConversionPattern<ckks::BootstrapOp> {
  ConvertBootstrapOp(mlir::MLIRContext *context)
      : OpConversionPattern<ckks::BootstrapOp>(context) {}

  using OpConversionPattern<ckks::BootstrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, ckks::BootstrapOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<openfhe::BootstrapOp>(
        op, op.getOutput().getType(), cryptoContext, adaptor.getInput());
    return success();
  }
};

struct LWEToOpenfhe : public impl::LWEToOpenfheBase<LWEToOpenfhe> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToOpenfheTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<openfhe::OpenfheDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target.addIllegalDialect<ckks::CKKSDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();
    // We can keep the following ops, which the emitter can handle directly
    target.addLegalOp<lwe::ReinterpretUnderlyingTypeOp, lwe::RLWEDecodeOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg = op.getFunctionType().getNumInputs() > 0 &&
                                 mlir::isa<openfhe::CryptoContextType>(
                                     *op.getFunctionType().getInputs().begin());
      auto containsCryptoOps =
          containsDialects<lwe::LWEDialect, bgv::BGVDialect, ckks::CKKSDialect>(
              op);
      auto containsCryptoArg =
          containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect,
                                    ckks::CKKSDialect>(op);
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!(containsCryptoOps || containsCryptoArg) || hasCryptoContextArg);
    });

    // Ensures that callee function signature is consistent
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      auto operandTypes = op.getCalleeType().getInputs();
      auto containsCryptoArg = llvm::any_of(operandTypes, [&](Type argType) {
        return DialectEqual<lwe::LWEDialect, bgv::BGVDialect,
                            ckks::CKKSDialect>()(&argType.getDialect());
      });
      auto hasCryptoContextArg =
          !operandTypes.empty() &&
          mlir::isa<openfhe::CryptoContextType>(*operandTypes.begin());
      return (!containsCryptoArg || hasCryptoContextArg);
    });

    patterns.add<
        /////////////////////
        // LWE Op Patterns //
        /////////////////////

        // Update Func Op Signature
        AddCryptoContextArg,

        // Update Func CallOp Signature
        ConvertFuncCallOp,

        // Handle LWE encode and en/decrypt
        // Note: `lwe.decode` is handled directly by the OpenFHE emitter
        ConvertEncodeOp, ConvertEncryptOp, ConvertDecryptOp,

        // Scheme-agnostic RLWE Arithmetic Ops:
        ConvertLWEBinOp<lwe::RAddOp, openfhe::AddOp>,
        ConvertLWEBinOp<lwe::RSubOp, openfhe::SubOp>,
        ConvertLWEBinOp<lwe::RMulOp, openfhe::MulNoRelinOp>,
        ConvertUnaryOp<lwe::RNegateOp, openfhe::NegateOp>,

        ///////////////////////////////////
        // Scheme-Specific Op Patterns   //
        ///////////////////////////////////
        // The Add/(Sub)/Mul-Plain ops are not really scheme-specific,
        // but do not currently have an analogue in the LWE dialect.
        // TODO (#1193): Extend "common lwe" to support ctxt-ptxt ops

        // AddPlain
        ConvertCiphertextPlaintextOp<bgv::AddPlainOp, openfhe::AddPlainOp>,
        ConvertCiphertextPlaintextOp<ckks::AddPlainOp, openfhe::AddPlainOp>,

        // TODO (#1200): SubPlain support for OpenFHE

        // MulPlain
        ConvertCiphertextPlaintextOp<bgv::MulPlainOp, openfhe::MulPlainOp>,
        ConvertCiphertextPlaintextOp<ckks::MulPlainOp, openfhe::MulPlainOp>,

        // Rotate
        ConvertRotateOp<bgv::RotateOp, openfhe::RotOp>,
        ConvertRotateOp<ckks::RotateOp, openfhe::RotOp>,
        // Relin
        ConvertRelinOp<bgv::RelinearizeOp, openfhe::RelinOp>,
        ConvertRelinOp<ckks::RelinearizeOp, openfhe::RelinOp>,
        // Modulus Switch (BGV only)
        lwe::ConvertModulusSwitchOp<bgv::ModulusSwitchOp>,
        // Rescale (CKKS version of Modulus Switch)
        lwe::ConvertModulusSwitchOp<ckks::RescaleOp>,
        // Bootstrap (CKKS only)
        ConvertBootstrapOp
        // End of Pattern List
        >(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::lwe
