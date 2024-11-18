#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DEF_MODARITHTOARITH
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

IntegerType convertModArithType(ModArithType type) {
  APInt modulus = type.getModulus().getValue();
  return IntegerType::get(type.getContext(), modulus.getBitWidth());
}

Type convertModArithLikeType(ShapedType type) {
  if (auto modArithType = llvm::dyn_cast<ModArithType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertModArithType(modArithType));
  }
  return type;
}

class ModArithToArithTypeConverter : public TypeConverter {
 public:
  ModArithToArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](ModArithType type) -> Type { return convertModArithType(type); });
    addConversion(
        [](ShapedType type) -> Type { return convertModArithLikeType(type); });
  }
};

// A helper function to generate the attribute or type
// needed to represent the result of mod_arith op as an integer
// before applying a remainder operation
template <typename Op>
TypedAttr modulusAttr(Op op, bool mul = false,
                      std::optional<APInt> constant = std::nullopt) {
  auto type = op.getResult().getType();
  auto modArithType = getResultModArithType(op);
  APInt modulus = modArithType.getModulus().getValue();

  auto width = modulus.getBitWidth();
  if (mul) {
    width *= 2;
  }

  auto intType = IntegerType::get(op.getContext(), width);
  APInt intValue = constant == std::nullopt ? modulus : *constant;
  auto truncmod = intValue.zextOrTrunc(width);

  if (auto st = mlir::dyn_cast<ShapedType>(type)) {
    auto containerType = st.cloneWith(st.getShape(), intType);
    return DenseElementsAttr::get(containerType, truncmod);
  }
  return IntegerAttr::get(intType, truncmod);
}

// used for extui/trunci
template <typename Op>
inline Type modulusType(Op op, bool mul = false) {
  return modulusAttr(op, mul).getType();
}

struct ConvertEncapsulate : public OpConversionPattern<EncapsulateOp> {
  ConvertEncapsulate(mlir::MLIRContext *context)
      : OpConversionPattern<EncapsulateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EncapsulateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertExtract : public OpConversionPattern<ExtractOp> {
  ConvertExtract(mlir::MLIRContext *context)
      : OpConversionPattern<ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto constOp = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                      op.getValue().getValue());
    rewriter.replaceOp(op, constOp);
    return success();
  }
};

struct ConvertReduce : public OpConversionPattern<ReduceOp> {
  ConvertReduce(mlir::MLIRContext *context)
      : OpConversionPattern<ReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    // ModArithType ensures cmod can be correctly interpreted as a signed number
    auto rems = b.create<arith::RemSIOp>(adaptor.getOperands()[0], cmod);
    auto add = b.create<arith::AddIOp>(rems, cmod);
    // TODO(#710): better with a subifge
    auto remu = b.create<arith::RemUIOp>(add, cmod);
    rewriter.replaceOp(op, remu);
    return success();
  }
};

// It is assumed inputs are canonical representatives
// ModArithType ensures add/sub result can not overflow
struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto add = b.create<arith::AddIOp>(adaptor.getLhs(), adaptor.getRhs());
    auto remu = b.create<arith::RemUIOp>(add, cmod);

    rewriter.replaceOp(op, remu);
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  ConvertSub(mlir::MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto sub = b.create<arith::SubIOp>(adaptor.getLhs(), adaptor.getRhs());
    auto add = b.create<arith::AddIOp>(sub, cmod);
    auto remu = b.create<arith::RemUIOp>(add, cmod);

    rewriter.replaceOp(op, remu);
    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(mlir::MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto lhs =
        b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getLhs());
    auto rhs =
        b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getRhs());
    auto mul = b.create<arith::MulIOp>(lhs, rhs);

    auto modArithType = getResultModArithType(op);
    // here mul in [0, q^2], internal constraint
    auto bar = b.create<mod_arith::BarrettReduceOp>(modArithType, mul);
    auto subifge = b.create<mod_arith::SubIfGEOp>(modArithType, bar);

    rewriter.replaceOp(op, subifge);
    return success();
  }
};

struct ConvertMac : public OpConversionPattern<MacOp> {
  ConvertMac(mlir::MLIRContext *context)
      : OpConversionPattern<MacOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MacOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op, true));
    auto x = b.create<arith::ExtUIOp>(modulusType(op, true),
                                      adaptor.getOperands()[0]);
    auto y = b.create<arith::ExtUIOp>(modulusType(op, true),
                                      adaptor.getOperands()[1]);
    auto acc = b.create<arith::ExtUIOp>(modulusType(op, true),
                                        adaptor.getOperands()[2]);
    auto mul = b.create<arith::MulIOp>(x, y);
    auto add = b.create<arith::AddIOp>(mul, acc);
    auto remu = b.create<arith::RemUIOp>(add, cmod);
    auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);

    rewriter.replaceOp(op, trunc);
    return success();
  }
};

struct ConvertBarrettReduce : public OpConversionPattern<BarrettReduceOp> {
  ConvertBarrettReduce(mlir::MLIRContext *context)
      : OpConversionPattern<BarrettReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BarrettReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Compute B = 2^{2 * width} and ratio = floordiv(B / modulus)
    auto resultModArithType = getResultModArithType(op);
    APInt modulus = resultModArithType.getModulus().getValue();
    auto mulWidth = getElementTypeOrSelf(op.getInput().getType()).getIntOrFloatBitWidth();
    auto width = modulus.getBitWidth();
    assert(2 * width == mulWidth);
    mulWidth = mulWidth + width;  // mulwidth = 3 * width
    auto B = APInt(mulWidth, 1).shl(2 * width);
    auto truncmod = modulus.zextOrTrunc(mulWidth);
    auto barrettRatio = B.udiv(truncmod);

    Type mulType = IntegerType::get(op.getContext(), mulWidth);

    // Create our pre-computed constants

    TypedAttr ratioAttr, shiftAttr, modAttr;
    if (auto st = mlir::dyn_cast<ShapedType>(op.getResult().getType())) {
      auto containerType = st.cloneWith(st.getShape(), mulType);
      ratioAttr = DenseElementsAttr::get(containerType, barrettRatio);
      shiftAttr = DenseElementsAttr::get(containerType, APInt(mulWidth, 2 * width));
      modAttr = DenseElementsAttr::get(containerType, truncmod);
      mulType = containerType;
    } else {
      ratioAttr = IntegerAttr::get(mulType, barrettRatio);
      shiftAttr = IntegerAttr::get(mulType, APInt(mulWidth, 2 * width));
      modAttr = IntegerAttr::get(mulType, truncmod);
    }
  
    auto ratioValue = b.create<arith::ConstantOp>(mulType, ratioAttr);
    auto shiftValue = b.create<arith::ConstantOp>(mulType, shiftAttr);
    auto modValue = b.create<arith::ConstantOp>(mulType, modAttr);

    // Intermediate value will be in the range [0,p^3) so we need to extend to
    // 3 * width
    auto extendOp = b.create<arith::ExtUIOp>(mulType, adaptor.getInput());

    // Compute x - floordiv(x * ratio, B) * mod

    auto mulRatioOp = b.create<arith::MulIOp>(extendOp, ratioValue);
    auto shrOp = b.create<arith::ShRUIOp>(mulRatioOp, shiftValue);    
    auto mulModOp = b.create<arith::MulIOp>(shrOp, modValue);
    auto subOp = b.create<arith::SubIOp>(extendOp, mulModOp);

    auto truncOp = b.create<arith::TruncIOp>(modulusType(op, false), subOp);

    rewriter.replaceOp(op, truncOp);
    return success();
  }
};

struct ConvertSubIfGE : public OpConversionPattern<SubIfGEOp> {
  ConvertSubIfGE(mlir::MLIRContext *context)
      : OpConversionPattern<SubIfGEOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubIfGEOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto input = adaptor.getInput();
    auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto sub = b.create<arith::SubIOp>(input, cmod);
    auto cmp = b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, input, cmod);
    auto select = b.create<arith::SelectOp>(cmp, sub, input);

    rewriter.replaceOp(op, select);
    return success();
  }
};

struct ModArithToArith : impl::ModArithToArithBase<ModArithToArith> {
  using ModArithToArithBase::ModArithToArithBase;

  void runOnOperation() override;
};

void ModArithToArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ModArithToArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<ModArithDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(context);
  patterns.add<
      ConvertEncapsulate, ConvertExtract, ConvertReduce, ConvertAdd, ConvertSub,
      ConvertMul, ConvertMac, ConvertBarrettReduce, ConvertSubIfGE,
      ConvertConstant, ConvertAny<>, ConvertAny<affine::AffineForOp>,
      ConvertAny<affine::AffineYieldOp>, ConvertAny<linalg::GenericOp> >(
      typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      tensor::EmptyOp, tensor::ExtractOp, tensor::InsertOp, affine::AffineForOp,
      affine::AffineYieldOp, linalg::GenericOp, linalg::YieldOp,
      tensor::ExtractSliceOp, tensor::InsertSliceOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir


