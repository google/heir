#include "include/Conversion/BGVToOpenfhe/BGVToOpenfhe.h"

#include <cassert>
#include <utility>

#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/BGV/IR/BGVOps.h"
#include "include/Dialect/LWE/IR/LWEAttributes.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "include/Dialect/Openfhe/IR/OpenfheOps.h"
#include "include/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Conversion/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOOPENFHE
#include "include/Conversion/BGVToOpenfhe/BGVToOpenfhe.h.inc"

class ToLWECiphertextTypeConverter : public TypeConverter {
 public:
  ToLWECiphertextTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
  }
};

bool containsBGVOps(func::FuncOp func) {
  auto walkResult = func.walk([&](Operation *op) {
    if (llvm::isa<bgv::BGVDialect>(op->getDialect()))
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
  if (!cryptoContext.getType().isa<openfhe::CryptoContextType>()) {
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
    rewriter.replaceOp(
        op, rewriter.create<OpenfheUnaryOp>(op.getLoc(), cryptoContext,
                                            adaptor.getOperands()[0]));
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
    rewriter.replaceOp(op,
                       rewriter.create<OpenfheBinOp>(op.getLoc(), cryptoContext,
                                                     adaptor.getOperands()[0],
                                                     adaptor.getOperands()[1]));
    return success();
  }
};

using ConvertNegateOp = ConvertUnaryOp<Negate, openfhe::NegateOp>;

using ConvertAddOp = ConvertBinOp<AddOp, openfhe::AddOp>;
using ConvertSubOp = ConvertBinOp<SubOp, openfhe::SubOp>;
using ConvertMulOp = ConvertBinOp<MulOp, openfhe::MulNoRelinOp>;

struct ConvertRotateOp : public OpConversionPattern<Rotate> {
  ConvertRotateOp(mlir::MLIRContext *context)
      : OpConversionPattern<Rotate>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Rotate op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    Value castOffset =
        llvm::TypeSwitch<Type, Value>(adaptor.getOffset().getType())
            .Case<IndexType>([&](auto ty) {
              return rewriter
                  .create<arith::IndexCastOp>(
                      op.getLoc(), rewriter.getI64Type(), adaptor.getOffset())
                  .getResult();
            })
            .Case<IntegerType>([&](IntegerType ty) {
              if (ty.getWidth() < 64) {
                return rewriter
                    .create<arith::ExtUIOp>(op.getLoc(), rewriter.getI64Type(),
                                            adaptor.getOffset())
                    .getResult();
              }
              return rewriter
                  .create<arith::TruncIOp>(op.getLoc(), rewriter.getI64Type(),
                                           adaptor.getOffset())
                  .getResult();
            });
    rewriter.replaceOp(
        op, rewriter.create<openfhe::RotOp>(op.getLoc(), cryptoContext,
                                            adaptor.getInput(), castOffset));
    return success();
  }
};

bool checkRelinToBasis(llvm::ArrayRef<int> toBasis) {
  if (toBasis.size() != 2) return false;
  return toBasis[0] == 0 && toBasis[1] == 1;
}
struct ConvertRelinOp : public OpConversionPattern<Relinearize> {
  ConvertRelinOp(mlir::MLIRContext *context)
      : OpConversionPattern<Relinearize>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Relinearize op, OpAdaptor adaptor,
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
    rewriter.replaceOp(op, rewriter.create<openfhe::RelinOp>(
                               op.getLoc(), cryptoContext, adaptor.getInput()));
    return success();
  }
};

struct ConvertModulusSwitchOp : public OpConversionPattern<ModulusSwitch> {
  ConvertModulusSwitchOp(mlir::MLIRContext *context)
      : OpConversionPattern<ModulusSwitch>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModulusSwitch op, OpAdaptor adaptor,
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

struct BGVToOpenfhe : public impl::BGVToOpenfheBase<BGVToOpenfhe> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToLWECiphertextTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<openfhe::OpenfheDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg = op.getFunctionType().getNumInputs() > 0 &&
                                 op.getFunctionType()
                                     .getInputs()
                                     .begin()
                                     ->isa<openfhe::CryptoContextType>();
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsBGVOps(op) || hasCryptoContextArg);
    });
    patterns.add<AddCryptoContextArg, ConvertAddOp, ConvertSubOp, ConvertMulOp,
                 ConvertNegateOp, ConvertRotateOp, ConvertRelinOp,
                 ConvertModulusSwitchOp>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
