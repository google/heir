#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
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

    if (isa<IndexType>(op.getValue().getType())) {
      return failure();
    }

    auto result = b.create<mod_arith::ConstantOp>(mod_arith::ModArithAttr::get(
        convertArithType(op.getType()),
        cast<IntegerAttr>(op.getValue()).getValue().getSExtValue()));

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertExt : public OpConversionPattern<mlir::arith::ExtSIOp> {
  ConvertExt(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtSIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ExtSIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<mod_arith::ModSwitchOp>(
        op.getLoc(), convertArithType(op.getType()), op.getIn());
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

    // Check if origin is memref.load -> Add encapsulate
    auto lhsDefOp = op.getLhs().template getDefiningOp<memref::LoadOp>();
    auto rhsDefOp = op.getRhs().template getDefiningOp<memref::LoadOp>();

    auto lhsOp = adaptor.getLhs();
    auto rhsOp = adaptor.getRhs();

    if (lhsDefOp) {
      lhsOp = b.create<mod_arith::EncapsulateOp>(
          convertArithType(lhsDefOp.getType()), lhsDefOp.getResult());
    }
    if (rhsDefOp) {
      rhsOp = b.create<mod_arith::EncapsulateOp>(
          convertArithType(rhsDefOp.getType()), rhsDefOp.getResult());
    }

    auto result = b.create<TargetModArithOp>(lhsOp, rhsOp);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <class Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

struct ConvertMemRefLoad : public OpConversionPattern<mlir::memref::LoadOp> {
  ConvertMemRefLoad(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::memref::LoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // auto *parent = op.getMemRef().getDefiningOp();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, retTypes, adaptor.getOperands(), op->getAttrs());

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

  target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
      [](mlir::arith::ConstantOp op) {
        return isa<IndexType>(op.getValue().getType());
      });

  target.addDynamicallyLegalOp<memref::LoadOp>([&](Operation *op) {
    return cast<memref::LoadOp>(op).getMemRef().getDefiningOp();
    // auto elementType = cast<memref::LoadOp>(op).getMemRef().getDefiningOp();
  });

  // target.addDynamicallyLegalOp<memref::AllocOp, memref::DeallocOp,
  //                              memref::StoreOp, memref::GlobalOp,
  //                              memref::GetGlobalOp,
  //                              memref::SubViewOp, memref::CopyOp,
  //                              tensor::FromElementsOp, tensor::ExtractOp>(
  //     [&](Operation *op) {
  //       return typeConverter.isLegal(op->getOperandTypes()) &&
  //              typeConverter.isLegal(op->getResultTypes());
  //     });

  RewritePatternSet patterns(context);
  patterns.add<
      ConvertConstant, ConvertExt, ConvertMemRefLoad,
      ConvertBinOp<mlir::arith::AddIOp, mod_arith::AddOp>,
      ConvertBinOp<mlir::arith::SubIOp, mod_arith::SubOp>,
      ConvertBinOp<mlir::arith::MulIOp, mod_arith::MulOp>,
      GenericOpPattern<memref::LoadOp>
      // GenericOpPattern<memref::AllocOp>, GenericOpPattern<memref::DeallocOp>,
      // GenericOpPattern<memref::StoreOp>, GenericOpPattern<memref::SubViewOp>,
      // GenericOpPattern<memref::CopyOp>,
      // GenericOpPattern<tensor::FromElementsOp>,
      // GenericOpPattern<tensor::ExtractOp>

      >(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace arith
}  // namespace heir
}  // namespace mlir
