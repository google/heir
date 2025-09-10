#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"

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
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOOPENFHE
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h.inc"

ToOpenfheTypeConverter::ToOpenfheTypeConverter(MLIRContext* ctx) {
  addConversion([](Type type) { return type; });
  addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
    return openfhe::PublicKeyType::get(ctx);
  });
  addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
    return openfhe::PrivateKeyType::get(ctx);
  });
}

FailureOr<Value> getContextualCryptoContext(Operation* op) {
  auto result = getContextualArgFromFunc<openfhe::CryptoContextType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found LWE op in a function without a crypto context argument."
              " Did the AddCryptoContextArg pattern fail to run?";
  }
  return result.value();
}

namespace {
// NOTE: we can not use containsDialect
// for FuncOp declaration, which does not have a body
template <typename... Dialects>
bool containsArgumentOfDialect(func::FuncOp funcOp) {
  return llvm::any_of(funcOp.getArgumentTypes(), [&](Type argType) {
    return DialectEqual<Dialects...>()(&argType.getDialect());
  });
}

struct AddCryptoContextArg : public OpConversionPattern<func::FuncOp> {
  AddCryptoContextArg(mlir::MLIRContext* context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
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
    rewriter.modifyOpInPlace(op, [&] {
      (void)op.insertArgument(0, cryptoContextType, nullptr, op.getLoc());
    });

    return success();
  }
};

struct ConvertFuncCallOp : public OpConversionPattern<func::CallOp> {
  ConvertFuncCallOp(mlir::MLIRContext* context)
      : OpConversionPattern<func::CallOp>(context) {}

  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
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

    rewriter
        .replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes, newOperands)
        ->setDialectAttrs(op->getDialectAttrs());
    return success();
  }
};

struct ConvertEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  ConvertEncryptOp(mlir::MLIRContext* context)
      : OpConversionPattern<lwe::RLWEEncryptOp>(context) {}

  using OpConversionPattern<lwe::RLWEEncryptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, typename lwe::RLWEEncryptOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(
        op, openfhe::EncryptOp::create(rewriter, op.getLoc(),
                                       op.getOutput().getType(), cryptoContext,
                                       adaptor.getInput(), adaptor.getKey()));
    return success();
  }
};

struct ConvertDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  ConvertDecryptOp(mlir::MLIRContext* context)
      : OpConversionPattern<RLWEDecryptOp>(context) {}

  using OpConversionPattern<RLWEDecryptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RLWEDecryptOp op, RLWEDecryptOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(
        op, openfhe::DecryptOp::create(
                rewriter, op.getLoc(), op.getOutput().getType(), cryptoContext,
                adaptor.getInput(), adaptor.getSecretKey()));
    return success();
  }
};

struct ConvertEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  explicit ConvertEncodeOp(const mlir::TypeConverter& typeConverter,
                           mlir::MLIRContext* context)
      : mlir::OpConversionPattern<lwe::RLWEEncodeOp>(typeConverter, context) {}

  // OpenFHE has a convention that all inputs to MakePackedPlaintext are
  // std::vector<int64_t>, so we need to cast the input to that type.
  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;
    Value cryptoContext = result.value();

    Value input = adaptor.getInput();
    auto elementTy = getElementTypeOrSelf(input.getType());
    auto tensorTy = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!tensorTy) {
      return op.emitError() << "Expected a tensor type for input; maybe "
                               "assign_layout wasn't properly lowered?";
    }

    // Cast inputs to the correct types for OpenFHE API.
    if (auto intTy = mlir::dyn_cast<IntegerType>(elementTy)) {
      if (intTy.getWidth() > 64)
        return op.emitError() << "No supported packing technique for integers "
                                 "bigger than 64 bits.";
      // OpenFHE has a convention that all inputs to MakePackedPlaintext are
      // std::vector<int64_t>, so we need to cast the input to that type.
      if (intTy.getWidth() < 64) {
        auto int64Ty = rewriter.getIntegerType(64);
        auto newTensorTy = RankedTensorType::get(tensorTy.getShape(), int64Ty);
        if (intTy.getWidth() == 1) {
          // Sign extending an i1 results in a -1 i64, so ensure that booleans
          // are unsigned.
          input =
              arith::ExtUIOp::create(rewriter, op.getLoc(), newTensorTy, input);
        } else {
          input =
              arith::ExtSIOp::create(rewriter, op.getLoc(), newTensorTy, input);
        }
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
        input =
            arith::ExtFOp::create(rewriter, op.getLoc(), newTensorTy, input);
      }
    }

    lwe::LWEPlaintextType plaintextType = op.getResult().getType();
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
  ConvertBootstrapOp(mlir::MLIRContext* context)
      : OpConversionPattern<ckks::BootstrapOp>(context) {}

  using OpConversionPattern<ckks::BootstrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, ckks::BootstrapOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<openfhe::BootstrapOp>(
        op, op.getOutput().getType(), cryptoContext, adaptor.getInput());
    return success();
  }
};
}  // namespace

struct LWEToOpenfhe : public impl::LWEToOpenfheBase<LWEToOpenfhe> {
  // See https://github.com/llvm/llvm-project/pull/127772
  // During dialect conversion, the attribute of the func::CallOp is not
  // preserved. We save the dialect attributes of func::CallOp before
  // conversion and restore them after conversion.
  //
  // Note that this is not safe as after conversion the order of func::CallOp
  // may change. However, this is the best we can do for now as we do not have
  // a map from the old func::CallOp to the new func::CallOp.
  SmallVector<SmallVector<NamedAttribute>> funcCallOpDialectAttrs;

  void saveFuncCallOpDialectAttrs() {
    funcCallOpDialectAttrs.clear();
    auto* module = getOperation();
    module->walk([&](func::CallOp callOp) {
      SmallVector<NamedAttribute> dialectAttrs;
      for (auto namedAttr : callOp->getDialectAttrs()) {
        dialectAttrs.push_back(namedAttr);
      }
      funcCallOpDialectAttrs.push_back(dialectAttrs);
    });
  }

  void restoreFuncCallOpDialectAttrs() {
    auto* module = getOperation();
    auto* funcCallOpDialectAttrsIter = funcCallOpDialectAttrs.begin();
    module->walk([&](func::CallOp callOp) {
      callOp->setDialectAttrs(*funcCallOpDialectAttrsIter);
      ++funcCallOpDialectAttrsIter;
    });
  }

  void runOnOperation() override {
    // Save the dialect attributes of func::CallOp before conversion.
    saveFuncCallOpDialectAttrs();

    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ToOpenfheTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<openfhe::OpenfheDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target.addIllegalDialect<ckks::CKKSDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();
    // We can keep the following ops, which the emitter can handle directly
    target.addLegalOp<lwe::ReinterpretApplicationDataOp, lwe::RLWEDecodeOp>();

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
        ConvertCiphertextPlaintextOp<lwe::RAddPlainOp, openfhe::AddPlainOp>,
        ConvertCiphertextPlaintextOp<lwe::RSubPlainOp, openfhe::SubPlainOp>,
        ConvertCiphertextPlaintextOp<lwe::RMulPlainOp, openfhe::MulPlainOp>,

        ///////////////////////////////////
        // Scheme-Specific Op Patterns   //
        ///////////////////////////////////

        // AddPlain
        ConvertCiphertextPlaintextOp<lwe::RAddPlainOp, openfhe::AddPlainOp>,

        // SubPlain
        ConvertCiphertextPlaintextOp<lwe::RSubPlainOp, openfhe::SubPlainOp>,

        // MulPlain
        ConvertCiphertextPlaintextOp<lwe::RMulPlainOp, openfhe::MulPlainOp>,

        // Rotate
        ConvertRotateOp<bgv::RotateColumnsOp, openfhe::RotOp>,
        ConvertRotateOp<ckks::RotateOp, openfhe::RotOp>,
        // Relin
        ConvertRelinOp<bgv::RelinearizeOp, openfhe::RelinOp>,
        ConvertRelinOp<ckks::RelinearizeOp, openfhe::RelinOp>,
        // Modulus Switch (BGV only)
        lwe::ConvertModulusSwitchOp<bgv::ModulusSwitchOp>,
        // Rescale (CKKS version of Modulus Switch)
        lwe::ConvertModulusSwitchOp<ckks::RescaleOp>,
        // Level Reduce
        lwe::ConvertLevelReduceOp<bgv::LevelReduceOp>,
        lwe::ConvertLevelReduceOp<ckks::LevelReduceOp>,
        // Bootstrap (CKKS only)
        ConvertBootstrapOp
        // End of Pattern List
        >(typeConverter, context);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      return signalPassFailure();
    }

    // Restore the dialect attributes of func::CallOp after conversion.
    restoreFuncCallOpDialectAttrs();
  }
};

}  // namespace mlir::heir::lwe
