#include "lib/Dialect/ModArith/Conversions/ArithToModArith/ArithToModArith.h"

#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "arith-to-mod-arith"

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DEF_ARITHTOMODARITH
#include "lib/Dialect/ModArith/Conversions/ArithToModArith/ArithToModArith.h.inc"

static ModArithType convertArithType(IntegerType type) {
  auto modulus = type.getIntOrFloatBitWidth();
  return ModArithType::get(type.getContext(),
                           mlir::IntegerAttr::get(type, modulus));
}

// ModArithType convertModArithLikeType(ShapedType type) {
//   if (auto modArithType =
//   llvm::dyn_cast<ModArithType>(type.getElementType())) {
//     return type.cloneWith(type.getShape(),
//     convertModArithType(modArithType));
//   }
//   return type;
// }

class ArithToModArithTypeConverter : public TypeConverter {
 public:
  ArithToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](IntegerType type) -> ModArithType {
      return convertArithType(type);
    });
    // addConversion(
    //     [](ShapedType type) -> Type { return convertArithLikeType(type); });
  }
};

// A herlper function to generate the attribute or type
// needed to represent the result of modarith op as an integer
// before applying a remainder operation
template <typename Op>
TypedAttr modulusAttr(Op op, bool mul = false) {
  auto type = op.getResult().getType();
  auto modArithType = getResultModArithType(op);
  APInt modulus = modArithType.getModulus().getValue();

  auto width = modulus.getBitWidth();
  if (mul) {
    width *= 2;
  }

  auto intType = IntegerType::get(op.getContext(), width);
  auto truncmod = modulus.zextOrTrunc(width);

  if (auto st = mlir::dyn_cast<ShapedType>(type)) {
    auto containerType = st.cloneWith(st.getShape(), intType);
    return DenseElementsAttr::get(containerType, truncmod);
  }
  return IntegerAttr::get(intType, truncmod);
}

struct ConvertConstant : public OpConversionPattern<::mlir::arith::ConstantOp> {
  ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<::mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<mod_arith::ConstantOp>(adaptor.getOperands());
    rewriter.replaceOp(op, result);
    return success();
    return success();
  }
};

// // It is assumed inputs are canonical representatives
// // ModArithType ensures add/sub result can not overflow
struct ConvertToAdd : public OpConversionPattern<::mlir::arith::AddIOp> {
  ConvertToAdd(mlir::MLIRContext *context)
      : OpConversionPattern<::mlir::arith::AddIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::AddIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result =
        b.create<mod_arith::AddOp>(adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
    return success();
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
  // target.addLegalDialect<ModArithDialect>();
  // target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(context);
  patterns.add<ConvertToAdd, ConvertConstant>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
