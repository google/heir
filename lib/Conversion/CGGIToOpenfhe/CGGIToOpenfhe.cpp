#include "lib/Conversion/CGGIToOpenfhe/CGGIToOpenfhe.h"

#include <numeric>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOOPENFHE
#include "lib/Conversion/CGGIToOpenfhe/CGGIToOpenfhe.h.inc"

// Remove this class if no type conversions are necessary
class CGGIToOpenfheTypeConverter : public TypeConverter {
 public:
  CGGIToOpenfheTypeConverter(MLIRContext *ctx) {
    // addConversion([](Type type) { return type; });
    // // FIXME: implement, replace FooType with the type that needs
    // // to be converted or remove this class
    // addConversion([ctx](lwe::LWECiphertextType type) -> Type {
    //   return type;
    // });
  }
};

// Commented this out bc it throws a linker error since there's another one in CGGI -> TFHE Rust bool
// bool containsCGGIOps(func::FuncOp func) {
//   auto walkResult = func.walk([&](Operation *op) {
//     if (llvm::isa<cggi::CGGIDialect>(op->getDialect()))
//       return WalkResult::interrupt();
//     return WalkResult::advance();
//   });
//   return walkResult.wasInterrupted();
// }

// FIXME: I stole these two from the BGVToOpenfhe conversion; is there a better
// way to share code?
struct AddCryptoContextArg : public OpConversionPattern<func::FuncOp> {
  AddCryptoContextArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // if (!containsCGGIOps(op)) {
    //   return failure();
    // }

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

struct ConvertLutLincombOp : public OpConversionPattern<cggi::LutLinCombOp> {
  ConvertLutLincombOp(mlir::MLIRContext *context)
      : OpConversionPattern<cggi::LutLinCombOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cggi::LutLinCombOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto result = getContextualCryptoContext(op);
    if (failed(result)) return result;
    auto cryptoContext = result.value();

    auto inputs = op.getInputs();
    auto coefficients = op.getCoefficients();

    llvm::SmallVector<openfhe::MulConstOp, 4> preppedInputs;
    preppedInputs.reserve(coefficients.size());

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    for (int i = 0; i < coefficients.size(); i++) {
      preppedInputs.push_back(b.create<openfhe::MulConstOp>(
          cryptoContext, inputs[i],
          b.create<arith::ConstantOp>(b.getI32Type(),
                                      b.getI32IntegerAttr(coefficients[i]))
              .getResult()));
    }

    auto sum = b.create<openfhe::AddOp>(cryptoContext, preppedInputs[0],
                                        preppedInputs[1]);
    for (int i = 2; i < preppedInputs.size(); i++) {
      sum = b.create<openfhe::AddOp>(cryptoContext, sum, preppedInputs[i]);
    }
    rewriter.replaceOp(op, sum);

    return success();
  }
};

// struct ConvertTrivialEncryptOp
//     : public OpConversionPattern<lwe::TrivialEncryptOp> {
//   ConvertTrivialEncryptOp(mlir::MLIRContext *context)
//       : mlir::OpConversionPattern<lwe::TrivialEncryptOp>(context) {}

//   LogicalResult matchAndRewrite(
//       lwe::TrivialEncryptOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {

//     auto result = getContextualCryptoContext(op);
//     if (failed(result)) return result;
//     auto cryptoContext = result.value();

//     auto encodeOp = op.getInput().getDefiningOp<lwe::EncodeOp>();

//   }

// };

struct CGGIToOpenfhe : public impl::CGGIToOpenfheBase<CGGIToOpenfhe> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    CGGIToOpenfheTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<openfhe::OpenfheDialect, func::FuncDialect,
                           memref::MemRefDialect, lwe::LWEDialect,
                           cggi::CGGIDialect>();
    // target.addIllegalDialect<cggi::CGGIDialect>();
    patterns.add<AddCryptoContextArg, ConvertLutLincombOp>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir