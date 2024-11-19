#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "arith-to-mod-arith"

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_DEF_ARITHTOMODARITH
#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

static mod_arith::ModArithType convertArithType(Type type) {
  auto modulusBitSize = type.getIntOrFloatBitWidth();
  auto modulus = (int)(1 << (modulusBitSize - 1)) - 1;
  return mod_arith::ModArithType::get(type.getContext(),
                                      mlir::IntegerAttr::get(type, modulus));
}

static Type convertArithLikeType(ShapedType type) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertArithType(arithType));
  }
  return type;
}

class ArithToModArithTypeConverter : public TypeConverter {
 public:
  ArithToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](IntegerType type) -> mod_arith::ModArithType {
      return convertArithType(type);
    });
    addConversion(
        [](ShapedType type) -> Type { return convertArithLikeType(type); });
  }
};

struct ConvertConstant : public OpConversionPattern<mlir::arith::ConstantOp> {
  ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<mod_arith::ConstantOp>(mod_arith::ModArithAttr::get(
        convertArithType(op.getType()),
        cast<IntegerAttr>(op.getValue()).getValue().getSExtValue()));

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceArithOp, typename TargetModArithOp>
struct ConvertBinOp : public OpConversionPattern<SourceArithOp> {
  ConvertBinOp(mlir::MLIRContext *context)
      : OpConversionPattern<SourceArithOp>(context) {}

  using OpConversionPattern<SourceArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceArithOp op, typename SourceArithOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result =
        b.create<TargetModArithOp>(adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithToModArith : impl::ArithToModArithBase<ArithToModArith> {
  using ArithToModArithBase::ArithToModArithBase;

  void runOnOperation() override;
};

void ArithToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ArithToModArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addLegalDialect<mod_arith::ModArithDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect>();

  RewritePatternSet patterns(context);
  patterns
      .add<ConvertConstant, ConvertBinOp<mlir::arith::AddIOp, mod_arith::AddOp>,
           ConvertBinOp<mlir::arith::SubIOp, mod_arith::SubOp>,
           ConvertBinOp<mlir::arith::MulIOp, mod_arith::MulOp>>(typeConverter,
                                                                context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace arith
}  // namespace heir
}  // namespace mlir
