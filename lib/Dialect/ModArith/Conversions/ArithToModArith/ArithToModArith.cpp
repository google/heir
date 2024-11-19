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

ModArithType convertArithType(Type type) {
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
    addConversion([](ModArithType type) -> ModArithType {
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

// used for extui/trunci
template <typename Op>
inline Type modulusType(Op op, bool mul = false) {
  return modulusAttr(op, mul).getType();
}

// struct ConvertEncapsulate : public OpConversionPattern<EncapsulateOp> {
//   ConvertEncapsulate(mlir::MLIRContext *context)
//       : OpConversionPattern<EncapsulateOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       EncapsulateOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     rewriter.replaceAllUsesWith(op.getResult(), adaptor.getOperands()[0]);
//     rewriter.eraseOp(op);
//     return success();
//   }
// };

// struct ConvertExtract : public OpConversionPattern<ExtractOp> {
//   ConvertExtract(mlir::MLIRContext *context)
//       : OpConversionPattern<ExtractOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       ExtractOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     rewriter.replaceAllUsesWith(op.getResult(), adaptor.getOperands()[0]);
//     rewriter.eraseOp(op);
//     return success();
//   }
// };

// struct ConvertReduce : public OpConversionPattern<ReduceOp> {
//   ConvertReduce(mlir::MLIRContext *context)
//       : OpConversionPattern<ReduceOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       ReduceOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     ImplicitLocOpBuilder b(op.getLoc(), rewriter);

//     auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
//     // ModArithType ensures cmod can be correctly interpreted as a signed
//     number auto rems = b.create<arith::RemSIOp>(adaptor.getOperands()[0],
//     cmod); auto add = b.create<arith::AddIOp>(rems, cmod);
//     // TODO(#710): better with a subifge
//     auto remu = b.create<arith::RemUIOp>(add, cmod);
//     rewriter.replaceOp(op, remu);
//     return success();
//   }
// };

// // It is assumed inputs are canonical representatives
// // ModArithType ensures add/sub result can not overflow
struct ConvertToAdd : public OpConversionPattern<arith::MulIOp> {
  ConvertToAdd(mlir::MLIRContext *context)
      : OpConversionPattern<arith::MulIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::MulIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    LLVM_DEBUG({ llvm::dbgs() << "################### Found one  mult:\n"; });

    // auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
    auto add = b.create<AddOp>(adaptor.getLhs(), adaptor.getRhs());
    // auto remu = b.create<arith::RemUIOp>(add, cmod);

    rewriter.replaceOp(op, add);
    return success();
  }
};

// struct ConvertSub : public OpConversionPattern<SubOp> {
//   ConvertSub(mlir::MLIRContext *context)
//       : OpConversionPattern<SubOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       SubOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     ImplicitLocOpBuilder b(op.getLoc(), rewriter);

//     auto cmod = b.create<arith::ConstantOp>(modulusAttr(op));
//     auto sub = b.create<arith::SubIOp>(adaptor.getLhs(), adaptor.getRhs());
//     auto add = b.create<arith::AddIOp>(sub, cmod);
//     auto remu = b.create<arith::RemUIOp>(add, cmod);

//     rewriter.replaceOp(op, remu);
//     return success();
//   }
// };

struct ConvertToMAC : public OpConversionPattern<arith::AddIOp> {
  ConvertToMAC(mlir::MLIRContext *context)
      : OpConversionPattern<arith::AddIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::AddIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    LLVM_DEBUG({
      llvm::dbgs() << "################### Found one "
                   << op->getParentOp()->getName() << ":\n";
    });

    auto mulOp = mlir::dyn_cast<arith::MulIOp>(
        op->getParentOp());  // Trust on the canonalisation?

    if (!mulOp) return failure();

    // Create a new ArithMacOp
    auto macOp =
        b.create<heir::mod_arith::MacOp>(op.getType(), mulOp.getOperand(0),
                                         mulOp.getOperand(1), adaptor.getLhs());

    // Replace the old operations
    rewriter.replaceOp(op, macOp);
    return success();

    // auto cmod = b.create<arith::ConstantOp>(modulusAttr(op, true));
    // auto lhs =
    //     b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getLhs());
    // auto rhs =
    //     b.create<arith::ExtUIOp>(modulusType(op, true), adaptor.getRhs());
    // auto mul = b.create<arith::MulIOp>(lhs, rhs);
    // auto remu = b.create<arith::RemUIOp>(mul, cmod);
    // auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);
  }
};

// struct ConvertToMac : public OpConversionPattern<MacOp> {
//   ConvertToMac(mlir::MLIRContext *context)
//       : OpConversionPattern<MacOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       MacOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     ImplicitLocOpBuilder b(op.getLoc(), rewriter);

//     auto cmod = b.create<arith::ConstantOp>(modulusAttr(op, true));
//     auto x = b.create<arith::ExtUIOp>(modulusType(op, true),
//                                       adaptor.getOperands()[0]);
//     auto y = b.create<arith::ExtUIOp>(modulusType(op, true),
//                                       adaptor.getOperands()[1]);
//     auto acc = b.create<arith::ExtUIOp>(modulusType(op, true),
//                                         adaptor.getOperands()[2]);
//     auto mul = b.create<arith::MulIOp>(x, y);
//     auto add = b.create<arith::AddIOp>(mul, acc);
//     auto remu = b.create<arith::RemUIOp>(add, cmod);
//     auto trunc = b.create<arith::TruncIOp>(modulusType(op), remu);

//     rewriter.replaceOp(op, trunc);
//     return success();
//   }
// };

struct ArithToModArith : impl::ArithToModArithBase<ArithToModArith> {
  using ArithToModArithBase::ArithToModArithBase;

  void runOnOperation() override;
};

void ArithToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ArithToModArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addLegalDialect<ModArithDialect>();
  // target.addIllegalOp(arith::AddIOp()->getName());
  // target.addIllegalOp(arith::MulIOp()->getName());

  RewritePatternSet patterns(context);
  patterns.add<ConvertToMAC, ConvertToAdd>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
