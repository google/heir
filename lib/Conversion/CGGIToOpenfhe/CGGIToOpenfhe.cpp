#include "lib/Conversion/CGGIToOpenfhe/CGGIToOpenfhe.h"

#include <iostream>
#include <numeric>

// #include "lib/Conversion/Utils.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "cggi-to-openfhe"

namespace mlir::heir {

#define GEN_PASS_DEF_CGGITOOPENFHE
#include "lib/Conversion/CGGIToOpenfhe/CGGIToOpenfhe.h.inc"

// Remove this class if no type conversions are necessary
class CGGIToOpenfheTypeConverter : public TypeConverter {
 public:
  CGGIToOpenfheTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
  }
};

// Commented this out bc it throws a linker error since there's another one in
// CGGI -> TFHE Rust bool
static bool containsCGGIOps2(func::FuncOp func) {
  auto walkResult = func.walk([&](Operation *op) {
    if (llvm::isa<cggi::CGGIDialect>(op->getDialect()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

// FIXME: I stole these two from the BGVToOpenfhe conversion; is there a better
// way to share code?
struct AddCryptoContextParam : public OpConversionPattern<func::FuncOp> {
  AddCryptoContextParam(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "Attempting to add context param to: " << op << "\n");

    if (!containsCGGIOps2(op)) {
      LLVM_DEBUG(llvm::dbgs() << "No crypto ops, skipping...\n");
      return failure();
    }

    auto cryptoContextType = openfhe::BinFHEContextType::get(getContext());
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

namespace {
FailureOr<Value> getContextualCryptoContext(Operation *op) {
  Value cryptoContext = op->getParentOfType<func::FuncOp>()
                            .getBody()
                            .getBlocks()
                            .front()
                            .getArguments()
                            .front();
  if (!mlir::isa<openfhe::BinFHEContextType>(cryptoContext.getType())) {
    return op->emitOpError()
           << "Found CGGI op in a function without a public "
              "key argument. Did the AddCryptoContextArg pattern fail to run?";
  }
  return cryptoContext;
}
}  // namespace

struct AddCryptoContextArg : public OpConversionPattern<func::CallOp> {
  AddCryptoContextArg(mlir::MLIRContext *context)
      : OpConversionPattern<func::CallOp>(context, /* benefit= */ 2) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] {
      auto result = getContextualCryptoContext(op);
      if (failed(result)) return;
      auto context = result.value();
      op->insertOperands(0, {context});
    });

    return success();
  }
};

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

    llvm::SmallVector<openfhe::LWEMulConstOp, 4> preppedInputs;
    preppedInputs.reserve(coefficients.size());

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto scheme = b.create<openfhe::GetLWESchemeOp>(cryptoContext).getResult();

    for (int i = 0; i < coefficients.size(); i++) {
      preppedInputs.push_back(b.create<openfhe::LWEMulConstOp>(
          scheme, inputs[i],
          b.create<arith::ConstantOp>(b.getI64Type(),
                                      b.getI64IntegerAttr(coefficients[i]))
              .getResult()));
    }

    mlir::Value lutInput;

    if (preppedInputs.size() > 1) {
      openfhe::LWEAddOp sum = b.create<openfhe::LWEAddOp>(
          scheme, preppedInputs[0].getResult(), preppedInputs[1].getResult());

      for (int i = 2; i < preppedInputs.size(); i++) {
        sum = b.create<openfhe::LWEAddOp>(scheme, sum, preppedInputs[i]);
      }

      lutInput = sum.getResult();
    } else {
      lutInput = preppedInputs[0].getResult();
    }

    // now create the LUT
    // llvm::SmallSetVector<int, 8> lutBits;
    // auto lutAttr = op.getLookupTableAttr();
    // int width = coefficients.size();
    // for (int i = 0; i < (1 << width); i++) {
    //   int index = 0;
    //   for (int j = 0; j < width; j++) {
    //     index += ((i >> (width - 1 - j)) & 1) * coefficients[j];
    //   }
    //   while (index < 0) index += 8;
    //   if ((lutAttr.getValue().getZExtValue() >> i) & 1) lutBits.insert(index
    //   % 8);
    // }

    // Extract message indices from the integer LUT mask, LSb-first.
    llvm::SmallVector<int, 8> lutBits;
    auto lutAttr = op.getLookupTableAttr();
    uint64_t lutValue = lutAttr.getValue().getZExtValue();
    for (int i = 0; i < 8; i++) {
      if ((lutValue >> i) & 1ULL) {
        lutBits.push_back(i);
      }
    }

    auto makeLut = b.create<openfhe::MakeLutOp>(
        cryptoContext, b.getDenseI32ArrayAttr(lutBits));
    auto evalFunc = b.create<openfhe::EvalFuncOp>(
        lutInput.getType(), cryptoContext, makeLut.getResult(), lutInput);
    rewriter.replaceOp(op, evalFunc);
    return success();
  }
};

struct CGGIToOpenfhe : public impl::CGGIToOpenfheBase<CGGIToOpenfhe> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    CGGIToOpenfheTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addLegalDialect<openfhe::OpenfheDialect, memref::MemRefDialect,
                           lwe::LWEDialect>();

    target.addIllegalOp<cggi::LutLinCombOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp func) {
      bool hasCryptoContext = func.getFunctionType().getNumInputs() > 0 &&
                              mlir::isa<openfhe::BinFHEContextType>(
                                  *func.getFunctionType().getInputs().begin());
      return hasCryptoContext || func.getName().starts_with("internal_generic");
    });

    target.addDynamicallyLegalOp<func::CallOp>([](func::CallOp call) {
      bool hasCryptoContext = !call.getArgOperands().empty() &&
                              mlir::isa<openfhe::BinFHEContextType>(
                                  *call.getArgOperands().getType().begin());
      return hasCryptoContext;
    });

    // target.addIllegalDialect<cggi::CGGIDialect>();
    patterns
        .add<AddCryptoContextParam, AddCryptoContextArg, ConvertLutLincombOp>(
            typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
