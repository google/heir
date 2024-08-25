#include "lib/Conversion/BGVToOpenfhe/BGVToOpenfhe.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOOPENFHE
#include "lib/Conversion/BGVToOpenfhe/BGVToOpenfhe.h.inc"

class ToOpenfheTypeConverter : public TypeConverter {
 public:
  ToOpenfheTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::RLWEPublicKeyType type) -> Type {
      return openfhe::PublicKeyType::get(ctx);
    });
    addConversion([ctx](lwe::RLWESecretKeyType type) -> Type {
      return openfhe::PrivateKeyType::get(ctx);
    });
  }
};

bool containsBGVOps(func::FuncOp func) {
  auto walkResult = func.walk([&](Operation *op) {
    if (llvm::isa<bgv::BGVDialect>(op->getDialect()) ||
        llvm::isa<lwe::RLWEEncryptOp>(op) || llvm::isa<lwe::RLWEDecryptOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

FailureOr<Value> getContextualCryptoContext(Operation *op) {
  Value cryptoContext = op->getParentOfType<func::FuncOp>()
                            .getBody()
                            .getBlocks()
                            .front()
                            .getArguments()
                            .front();
  if (!mlir::isa<openfhe::CryptoContextType>(cryptoContext.getType())) {
    return op->emitOpError()
           << "Found BGV op in a function without a public "
              "key argument. Did the AddCryptoContextArg pattern fail to run?";
  }
  return cryptoContext;
}

struct AddCryptoContextArg : public OpConversionPattern<func::FuncOp> {
  AddCryptoContextArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!containsBGVOps(op)) {
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

      Block &block = op.getBody().getBlocks().front();
      block.insertArgument(&block.getArguments().front(), cryptoContextType,
                           op.getLoc());
    });

    return success();
  }
};

template <typename UnaryOp, typename OpenfheUnaryOp>
struct ConvertUnaryOp : public OpConversionPattern<UnaryOp> {
  using OpConversionPattern<UnaryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnaryOp op, typename UnaryOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, rewriter.create<OpenfheUnaryOp>(
                               op.getLoc(), cryptoContext, adaptor.getInput()));
    return success();
  }
};

template <typename BinOp, typename OpenfheBinOp>
struct ConvertBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<OpenfheBinOp>(op, op.getOutput().getType(),
                                              cryptoContext, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

using ConvertNegateOp = ConvertUnaryOp<NegateOp, openfhe::NegateOp>;

using ConvertAddOp = ConvertBinOp<AddOp, openfhe::AddOp>;
using ConvertSubOp = ConvertBinOp<SubOp, openfhe::SubOp>;
using ConvertMulOp = ConvertBinOp<MulOp, openfhe::MulNoRelinOp>;

template <typename BinOp, typename OpenfheBinOp>
struct ConvertCiphertextPlaintextOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<OpenfheBinOp>(
        op, op.getOutput().getType(), cryptoContext,
        adaptor.getCiphertextInput(), adaptor.getPlaintextInput());
    return success();
  }
};

using ConvertMulPlainOp =
    ConvertCiphertextPlaintextOp<MulPlainOp, openfhe::MulPlainOp>;

struct ConvertRotateOp : public OpConversionPattern<RotateOp> {
  ConvertRotateOp(mlir::MLIRContext *context)
      : OpConversionPattern<RotateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, rewriter.create<openfhe::RotOp>(
                               op.getLoc(), cryptoContext, adaptor.getInput(),
                               adaptor.getOffset()));
    return success();
  }
};

bool checkRelinToBasis(llvm::ArrayRef<int> toBasis) {
  if (toBasis.size() != 2) return false;
  return toBasis[0] == 0 && toBasis[1] == 1;
}

struct ConvertRelinOp : public OpConversionPattern<RelinearizeOp> {
  ConvertRelinOp(mlir::MLIRContext *context)
      : OpConversionPattern<RelinearizeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    auto toBasis = adaptor.getToBasis();

    // Since the `Relinearize()` function in OpenFHE relinearizes a ciphertext
    // to the lowest level (for (1,s)), the `to_basis` of `bgv.RelinOp` must be
    // [0,1].
    if (!checkRelinToBasis(toBasis)) {
      op.emitError() << "toBasis must be [0, 1], got [" << toBasis << "]";
      return failure();
    }

    Value cryptoContext = result.value();
    rewriter.replaceOpWithNewOp<openfhe::RelinOp>(
        op, op.getOutput().getType(), cryptoContext, adaptor.getInput());
    return success();
  }
};

struct ConvertModulusSwitchOp : public OpConversionPattern<ModulusSwitchOp> {
  ConvertModulusSwitchOp(mlir::MLIRContext *context)
      : OpConversionPattern<ModulusSwitchOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModulusSwitchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, rewriter.create<openfhe::ModReduceOp>(
                               op.getLoc(), op.getOutput().getType(),
                               cryptoContext, adaptor.getInput()));
    return success();
  }
};

struct ConvertEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  ConvertEncryptOp(mlir::MLIRContext *context)
      : OpConversionPattern<lwe::RLWEEncryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    auto keyType = dyn_cast<lwe::RLWEPublicKeyType>(op.getKey().getType());
    if (!keyType)
      return op.emitError()
             << "OpenFHE only supports public key encryption for BGV.";

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
      : OpConversionPattern<lwe::RLWEDecryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEDecryptOp op, OpAdaptor adaptor,
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

// Rewrite extract as a multiplication by a one-hot plaintext, followed by a
// rotate.
struct ConvertExtractOp : public OpConversionPattern<ExtractOp> {
  ConvertExtractOp(mlir::MLIRContext *context)
      : OpConversionPattern<ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    // Not-directly-constant offsets could be supported by using -sccp or
    // including a constant propagation analysis in this pass. A truly
    // non-constant extract op seems unlikely, given that most programs should
    // be using rotate instead of extractions, and that we mainly have extract
    // as a terminating op for IRs that must output a secret<scalar> type.
    auto offsetOp = adaptor.getOffset().getDefiningOp<arith::ConstantOp>();
    if (!offsetOp) {
      return op.emitError()
             << "Expected extract offset arg to be constant integer, found "
             << adaptor.getOffset();
    }
    auto offsetAttr = llvm::dyn_cast<IntegerAttr>(offsetOp.getValue());
    if (!offsetAttr) {
      return op.emitError()
             << "Expected extract offset arg to be constant integer, found "
             << adaptor.getOffset();
    }
    int64_t offset = offsetAttr.getInt();

    auto ctTy = op.getInput().getType();
    auto ring = ctTy.getRlweParams().getRing();
    auto degree = ring.getPolynomialModulus().getPolynomial().getDegree();
    auto elementTy =
        dyn_cast<IntegerType>(op.getOutput().getType().getUnderlyingType());
    if (!elementTy) {
      op.emitError() << "Expected extract op to extract scalar from tensor "
                        "type, but found input underlying type "
                     << op.getInput().getType().getUnderlyingType()
                     << " and output underlying type "
                     << op.getOutput().getType().getUnderlyingType();
    }
    auto tensorTy = RankedTensorType::get({degree}, elementTy);

    SmallVector<Attribute> oneHotCleartextAttrs;
    oneHotCleartextAttrs.reserve(degree);
    for (size_t i = 0; i < degree; ++i) {
      oneHotCleartextAttrs.push_back(
          rewriter.getIntegerAttr(elementTy, i == offset ? 1 : 0));
    }

    auto b = ImplicitLocOpBuilder(op->getLoc(), rewriter);
    auto oneHotCleartext =
        b.create<arith::ConstantOp>(
             tensorTy, DenseElementsAttr::get(tensorTy, oneHotCleartextAttrs))
            .getResult();
    auto plaintextTy = lwe::RLWEPlaintextType::get(
        op.getContext(), ctTy.getEncoding(), ring, tensorTy);
    auto oneHotPlaintext =
        b.create<lwe::RLWEEncodeOp>(plaintextTy, oneHotCleartext,
                                    ctTy.getEncoding(), ring)
            .getResult();
    auto plainMul =
        b.create<bgv::MulPlainOp>(adaptor.getInput(), oneHotPlaintext)
            .getResult();
    auto rotated = b.create<bgv::RotateOp>(plainMul, offsetAttr);
    // It might make sense to move this op to the add-client-interface pass,
    // but it also seems like an implementation detail of OpenFHE, and not part
    // of BGV generally.
    auto recast = b.create<lwe::ReinterpretUnderlyingTypeOp>(
                       op.getOutput().getType(), rotated.getResult())
                      .getResult();
    rewriter.replaceOp(op, recast);
    return success();
  }
};

struct ConvertEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern<lwe::RLWEEncodeOp>::OpConversionPattern;

  // OpenFHE has a convention that all inputs to MakePackedPlaintext are
  // std::vector<int64_t>, so we need to cast the input to that type.
  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto tensorTy =
        mlir::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    // TODO(#646): support scalar inputs to encode.
    if (!tensorTy) return op.emitOpError() << "Unsupported input type";

    auto elementTy = tensorTy.getElementType();
    auto intTy = mlir::dyn_cast<IntegerType>(elementTy);
    if (!intTy)
      return op.emitOpError() << "Input element type must be an integer type";

    if (intTy.getWidth() >= 64)
      return op.emitError() << "No supported packing technique for integers "
                               "bigger than 64 bits.";

    auto int64Ty = rewriter.getIntegerType(64);
    auto newTensorTy = RankedTensorType::get(tensorTy.getShape(), int64Ty);
    auto castedInput = rewriter.create<arith::ExtSIOp>(op.getLoc(), newTensorTy,
                                                       adaptor.getInput());

    rewriter.replaceOpWithNewOp<lwe::RLWEEncodeOp>(
        op, op.getType(), castedInput, adaptor.getEncoding(),
        adaptor.getRing());

    return success();
  }
};

struct BGVToOpenfhe : public impl::BGVToOpenfheBase<BGVToOpenfhe> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToOpenfheTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<openfhe::OpenfheDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target.addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg = op.getFunctionType().getNumInputs() > 0 &&
                                 mlir::isa<openfhe::CryptoContextType>(
                                     *op.getFunctionType().getInputs().begin());
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsBGVOps(op) || hasCryptoContextArg);
    });

    target.addDynamicallyLegalOp<lwe::RLWEEncodeOp>([](lwe::RLWEEncodeOp op) {
      if (auto tensorType =
              mlir::dyn_cast<RankedTensorType>(op.getInput().getType())) {
        if (auto intType =
                mlir::dyn_cast<IntegerType>(tensorType.getElementType())) {
          return intType.getWidth() == 64;
        }
      }
      return false;
    });
    patterns.add<AddCryptoContextArg, ConvertAddOp, ConvertSubOp, ConvertMulOp,
                 ConvertMulPlainOp, ConvertNegateOp, ConvertRotateOp,
                 ConvertRelinOp, ConvertModulusSwitchOp, ConvertExtractOp,
                 ConvertEncryptOp, ConvertDecryptOp, ConvertEncodeOp>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
