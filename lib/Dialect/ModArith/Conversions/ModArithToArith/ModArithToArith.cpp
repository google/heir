#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"     // from @llvm-project
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
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DEF_MODARITHTOARITH
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

namespace {

IntegerType convertModArithType(ModArithType type) {
  IntegerType modulusType = cast<IntegerType>(type.getModulus().getType());
  return IntegerType::get(type.getContext(), modulusType.getWidth());
}

Type convertRNSType(rns::RNSType type) {
  auto basisTypes = type.getBasisTypes();
  auto modulusType = cast<IntegerType>(
      cast<ModArithType>(basisTypes[0]).getModulus().getType());
  return RankedTensorType::get({(int64_t)basisTypes.size()}, modulusType);
}

Type convertModArithOrRNSLikeType(ShapedType type) {
  auto elementType = type.getElementType();
  if (auto modArithType = llvm::dyn_cast<ModArithType>(elementType)) {
    return type.cloneWith(type.getShape(), convertModArithType(modArithType));
  }
  if (auto rnsType = llvm::dyn_cast<rns::RNSType>(elementType)) {
    auto basisTypes = rnsType.getBasisTypes();
    auto modulusType = cast<IntegerType>(
        cast<ModArithType>(basisTypes[0]).getModulus().getType());
    std::vector<int64_t> shape = type.getShape();
    shape.push_back((int64_t)basisTypes.size());
    return RankedTensorType::get(shape, modulusType);
  }
  return type;
}

// A helper function to generate the attribute or type
// needed to represent the result of mod_arith op as an integer
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

// used for extui/trunci
template <typename Op>
inline Type modulusType(Op op, bool mul = false) {
  return modulusAttr(op, mul).getType();
}

}  // namespace

class ModArithToArithTypeConverter : public TypeConverter {
 public:
  ModArithToArithTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](ModArithType type) -> Type { return convertModArithType(type); });
    addConversion(
        [](rns::RNSType type) -> Type { return convertRNSType(type); });
    addConversion([](ShapedType type) -> Type {
      return convertModArithOrRNSLikeType(type);
    });
  }
};

struct ConvertEncapsulate : public OpConversionPattern<EncapsulateOp> {
  ConvertEncapsulate(mlir::MLIRContext* context)
      : OpConversionPattern<EncapsulateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EncapsulateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertExtract : public OpConversionPattern<ExtractOp> {
  ConvertExtract(mlir::MLIRContext* context)
      : OpConversionPattern<ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  ConvertConstant(mlir::MLIRContext* context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto constOp =
        arith::ConstantOp::create(rewriter, op.getLoc(), adaptor.getValue());
    rewriter.replaceOp(op, constOp);
    return success();
  }
};

struct ConvertReduce : public OpConversionPattern<ReduceOp> {
  ConvertReduce(mlir::MLIRContext* context)
      : OpConversionPattern<ReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = arith::ConstantOp::create(b, modulusAttr(op));
    // ModArithType ensures cmod can be correctly interpreted as a signed number
    auto rems = arith::RemSIOp::create(b, adaptor.getOperands()[0], cmod);
    auto add = arith::AddIOp::create(b, rems, cmod);
    // TODO(#710): better with a subifge
    auto remu = arith::RemUIOp::create(b, add, cmod);
    rewriter.replaceOp(op, remu);
    return success();
  }
};

// It is assumed inputs are canonical representatives
// ModArithType ensures add/sub result can not overflow
struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext* context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = arith::ConstantOp::create(b, modulusAttr(op));
    auto add = arith::AddIOp::create(b, adaptor.getLhs(), adaptor.getRhs());
    auto remu = arith::RemUIOp::create(b, add, cmod);

    rewriter.replaceOp(op, remu);
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  ConvertSub(mlir::MLIRContext* context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = arith::ConstantOp::create(b, modulusAttr(op));
    auto sub = arith::SubIOp::create(b, adaptor.getLhs(), adaptor.getRhs());
    auto add = arith::AddIOp::create(b, sub, cmod);
    auto remu = arith::RemUIOp::create(b, add, cmod);

    rewriter.replaceOp(op, remu);
    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(mlir::MLIRContext* context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = arith::ConstantOp::create(b, modulusAttr(op, true));
    auto lhs =
        arith::ExtUIOp::create(b, modulusType(op, true), adaptor.getLhs());
    auto rhs =
        arith::ExtUIOp::create(b, modulusType(op, true), adaptor.getRhs());
    auto mul = arith::MulIOp::create(b, lhs, rhs);
    auto remu = arith::RemUIOp::create(b, mul, cmod);
    auto trunc = arith::TruncIOp::create(b, modulusType(op), remu);

    rewriter.replaceOp(op, trunc);
    return success();
  }
};

struct ConvertMac : public OpConversionPattern<MacOp> {
  ConvertMac(mlir::MLIRContext* context)
      : OpConversionPattern<MacOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MacOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmod = arith::ConstantOp::create(b, modulusAttr(op, true));
    auto x = arith::ExtUIOp::create(b, modulusType(op, true),
                                    adaptor.getOperands()[0]);
    auto y = arith::ExtUIOp::create(b, modulusType(op, true),
                                    adaptor.getOperands()[1]);
    auto acc = arith::ExtUIOp::create(b, modulusType(op, true),
                                      adaptor.getOperands()[2]);
    auto mul = arith::MulIOp::create(b, x, y);
    auto add = arith::AddIOp::create(b, mul, acc);
    auto remu = arith::RemUIOp::create(b, add, cmod);
    auto trunc = arith::TruncIOp::create(b, modulusType(op), remu);

    rewriter.replaceOp(op, trunc);
    return success();
  }
};

struct ConvertModSwitch : public OpConversionPattern<ModSwitchOp> {
  ConvertModSwitch(mlir::MLIRContext* context)
      : OpConversionPattern<ModSwitchOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  auto getInteger(ImplicitLocOpBuilder* b, IntegerType integerType) const {
    return [=](const APInt& value) mutable {
      return b->create<arith::ConstantOp>(IntegerAttr::get(integerType, value));
    };
  }

  LogicalResult matchAndRewrite(
      ModSwitchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto inputType = op.getInput().getType();
    auto outputType = op.getOutput().getType();
    if (auto inputModArith = dyn_cast<ModArithType>(inputType)) {
      if (auto outputModArith = dyn_cast<ModArithType>(outputType)) {
        auto oldModulus = inputModArith.getModulus().getValue();
        auto newModulus = outputModArith.getModulus().getValue();
        auto oldWidth = oldModulus.getBitWidth();
        auto newWidth = newModulus.getBitWidth();
        auto width = std::max(oldWidth, newWidth);
        auto integerType = IntegerType::get(op.getContext(), width);
        auto intConst = getInteger(&b, integerType);
        oldModulus = oldModulus.zext(width);
        newModulus = newModulus.zext(width);
        Value input = adaptor.getInput();
        if (newWidth > oldWidth) {
          input = arith::ExtUIOp::create(b, integerType, input);
        }
        auto cmod = intConst(newModulus);
        auto remu = arith::RemUIOp::create(b, input, cmod);
        auto cmp = arith::CmpIOp::create(b, arith::CmpIPredicate::ugt,
                                         intConst(oldModulus.lshr(1)), input);
        auto diffConst = intConst(newModulus - oldModulus.urem(newModulus));
        auto remuAdd = arith::AddIOp::create(b, remu, diffConst);
        auto remuAddMod =
            arith::RemUIOp::create(b, remuAdd, intConst(newModulus));
        Value result = arith::SelectOp::create(b, cmp, remu, remuAddMod);
        if (newWidth < oldWidth) {
          auto outputType = IntegerType::get(op.getContext(), newWidth);
          result = arith::TruncIOp::create(b, outputType, result);
        }
        rewriter.replaceOp(op, result);
      } else if (auto outputRNS = dyn_cast<rns::RNSType>(outputType)) {
        auto modWidth = inputModArith.getModulus().getValue().getBitWidth();
        auto rnsWidth = cast<ModArithType>(outputRNS.getBasisTypes()[0])
                            .getModulus()
                            .getValue()
                            .getBitWidth();
        auto modintegerType = IntegerType::get(op.getContext(), modWidth);
        auto rnsIntegerType = IntegerType::get(op.getContext(), rnsWidth);
        auto intConst = getInteger(&b, modintegerType);
        std::vector<Value> rnsElements;
        for (auto basisType : outputRNS.getBasisTypes()) {
          auto modArithType = cast<ModArithType>(basisType);
          auto modulus = modArithType.getModulus().getValue();
          modulus = modulus.zext(modWidth);
          auto cmod = intConst(modulus);
          auto remu = arith::RemUIOp::create(b, adaptor.getInput(), cmod);
          auto trunci = arith::TruncIOp::create(b, rnsIntegerType, remu);
          rnsElements.push_back(trunci);
        }
        auto tensor =
            tensor::FromElementsOp::create(b, ValueRange(rnsElements));
        rewriter.replaceOp(op, tensor);
      } else {
        llvm_unreachable("Verifier should make sure this doesn't happen.");
      }
    } else if (auto inputRNS = dyn_cast<rns::RNSType>(inputType)) {
      auto outputModArith = cast<ModArithType>(outputType);
      auto bigModulus = outputModArith.getModulus().getValue();
      auto outputWidth = bigModulus.getBitWidth();
      // The width of each element in the RNS type.
      auto rnsWidth = cast<ModArithType>(inputRNS.getBasisTypes()[0])
                          .getModulus()
                          .getValue()
                          .getBitWidth();
      // The width of intermediate results: In the formula (from OpenFHE)
      // ```
      // Sigma(i = 0 --> t-1) { M(i) * qt/qi * [(qt/qi)^(-1) mod qi] } mod qt
      // ```
      // The width of `qt/qi` is bounded by `outputWidth`. The width of `M(i)`
      // is bounded by `rnsWidth`, and the value of `(qt/qi)^(-1) mod qi` is
      // bounded by `qi`, so its width is bounded by `rnsWidth`.
      // So the width of `M(i) * qt/qi * [(qt/qi)^(-1) mod qi]` is bounded by
      // `outputWidth + 2 * rnsWidth`.
      auto width = outputWidth + 2 * rnsWidth +
                   llvm::Log2_64_Ceil(inputRNS.getBasisTypes().size());
      bigModulus = bigModulus.zext(width);
      auto integerType = IntegerType::get(op.getContext(), width);
      auto intConst = getInteger(&b, integerType);
      auto bigModulusConst = intConst(bigModulus);
      Value sum = intConst(APInt::getZero(width));
      for (int64_t i = 0; i < inputRNS.getBasisTypes().size(); i++) {
        auto modArithType = cast<ModArithType>(inputRNS.getBasisTypes()[i]);
        auto modulus = modArithType.getModulus().getValue();
        modulus = modulus.zext(width);
        auto tmp = bigModulus.udiv(modulus);
        auto coeffConst =
            intConst(tmp * multiplicativeInverse(tmp.urem(modulus), modulus));
        auto index = arith::ConstantIndexOp::create(b, i);
        auto extract =
            tensor::ExtractOp::create(b, adaptor.getInput(), ValueRange{index});
        auto extui = arith::ExtUIOp::create(b, integerType, extract);
        auto mul = arith::MulIOp::create(b, extui, coeffConst);
        sum = arith::AddIOp::create(b, sum, mul);
      }
      auto remu = arith::RemUIOp::create(b, sum, bigModulusConst);
      auto resultType = IntegerType::get(op.getContext(), outputWidth);
      auto trunci = arith::TruncIOp::create(b, resultType, remu);
      rewriter.replaceOp(op, trunci);
    } else {
      llvm_unreachable("Verifier should make sure this doesn't happen.");
    }
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "lib/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.cpp.inc"
}  // namespace rewrites

struct ConvertBarrettReduce : public OpConversionPattern<BarrettReduceOp> {
  ConvertBarrettReduce(mlir::MLIRContext* context)
      : OpConversionPattern<BarrettReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BarrettReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Compute B = 4^{bitWidth} and ratio = floordiv(B / modulus)
    auto input = adaptor.getInput();
    auto mod = op.getModulus();
    auto bitWidth = (mod - 1).getActiveBits();
    mod = mod.trunc(3 * bitWidth);
    auto B = APInt(3 * bitWidth, 1).shl(2 * bitWidth);
    auto barrettRatio = B.udiv(mod);

    Type intermediateType = IntegerType::get(b.getContext(), 3 * bitWidth);

    // Create our pre-computed constants
    TypedAttr ratioAttr, shiftAttr, modAttr;
    if (auto tensorType = dyn_cast<RankedTensorType>(input.getType())) {
      tensorType = tensorType.clone(tensorType.getShape(), intermediateType);
      ratioAttr = DenseElementsAttr::get(tensorType, barrettRatio);
      shiftAttr =
          DenseElementsAttr::get(tensorType, APInt(3 * bitWidth, 2 * bitWidth));
      modAttr = DenseElementsAttr::get(tensorType, mod);
      intermediateType = tensorType;
    } else if (auto integerType = dyn_cast<IntegerType>(input.getType())) {
      ratioAttr = IntegerAttr::get(intermediateType, barrettRatio);
      shiftAttr =
          IntegerAttr::get(intermediateType, APInt(3 * bitWidth, 2 * bitWidth));
      modAttr = IntegerAttr::get(intermediateType, mod);
    }

    auto ratioValue = arith::ConstantOp::create(b, intermediateType, ratioAttr);
    auto shiftValue = arith::ConstantOp::create(b, intermediateType, shiftAttr);
    auto modValue = arith::ConstantOp::create(b, intermediateType, modAttr);

    // Intermediate value will be in the range [0,p^3) so we need to extend to
    // 3*bitWidth
    auto extendOp = arith::ExtUIOp::create(b, intermediateType, input);

    // Compute x - floordiv(x * ratio, B) * mod
    auto mulRatioOp = arith::MulIOp::create(b, extendOp, ratioValue);
    auto shrOp = arith::ShRUIOp::create(b, mulRatioOp, shiftValue);
    auto mulModOp = arith::MulIOp::create(b, shrOp, modValue);
    auto subOp = arith::SubIOp::create(b, extendOp, mulModOp);

    auto truncOp = arith::TruncIOp::create(b, input.getType(), subOp);

    rewriter.replaceOp(op, truncOp);

    return success();
  }
};

struct ModArithToArith : impl::ModArithToArithBase<ModArithToArith> {
  using ModArithToArithBase::ModArithToArithBase;

  void runOnOperation() override;
};

void ModArithToArith::runOnOperation() {
  MLIRContext* context = &getContext();
  ModuleOp module = getOperation();
  ModArithToArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<ModArithDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);
  patterns.add<
      ConvertEncapsulate, ConvertExtract, ConvertReduce, ConvertAdd, ConvertSub,
      ConvertMul, ConvertMac, ConvertModSwitch, ConvertBarrettReduce,
      ConvertConstant, ConvertAny<>, ConvertAny<affine::AffineForOp>,
      ConvertAny<affine::AffineYieldOp>, ConvertAny<linalg::GenericOp> >(
      typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      tensor::EmptyOp, tensor::ExtractOp, tensor::InsertOp, tensor::CastOp,
      affine::AffineForOp, affine::AffineYieldOp, linalg::GenericOp,
      linalg::YieldOp, tensor::ExtractSliceOp, tensor::InsertSliceOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(module, target, std::move(patterns),
                                    config))) {
    signalPassFailure();
  }
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
