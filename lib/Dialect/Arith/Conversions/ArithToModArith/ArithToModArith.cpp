#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_DEF_ARITHTOMODARITH
#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

static mod_arith::ModArithType convertArithType(Type type) {
  auto modulusBitSize = (int64_t)type.getIntOrFloatBitWidth();
  auto modulus = (1L << (modulusBitSize - 1L));
  auto newType = mlir::IntegerType::get(type.getContext(), modulusBitSize + 1);

  return mod_arith::ModArithType::get(type.getContext(),
                                      mlir::IntegerAttr::get(newType, modulus));
}

static Type convertArithLikeType(ShapedType type) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertArithType(arithType));
  }
  return type;
}

static Value buildLoadOps(OpBuilder &builder, Type resultTypes,
                          ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  auto loadOp = inputs[0].getDefiningOp<memref::LoadOp>();

  if (!loadOp) return {};

  auto *globaMemReflOp = loadOp.getMemRef().getDefiningOp();

  if (!globaMemReflOp) return {};

  return builder.create<mod_arith::EncapsulateOp>(
      loc, convertArithType(loadOp.getType()), loadOp.getResult());
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
    addTargetMaterialization(buildLoadOps);
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

struct ConvertExtSI : public OpConversionPattern<mlir::arith::ExtSIOp> {
  ConvertExtSI(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtSIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ExtSIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<mod_arith::ModSwitchOp>(
        op.getLoc(), convertArithType(op.getType()), adaptor.getIn());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertExtUI : public OpConversionPattern<mlir::arith::ExtUIOp> {
  ConvertExtUI(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<mod_arith::ModSwitchOp>(
        op.getLoc(), convertArithType(op.getType()), adaptor.getIn());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLoadOp : public OpConversionPattern<mlir::memref::LoadOp> {
  ConvertLoadOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::memref::LoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto defineOp = op.getMemRef().getDefiningOp();

    if (op.getMemRef().getDefiningOp()) {
      if (isa<memref::GetGlobalOp>(defineOp)) {
        // skip global memref
        return success();
      }
    }

    auto result = rewriter.create<memref::LoadOp>(
        op.getLoc(), convertArithType(op.getType()), adaptor.getOperands()[0],
        op.getIndices());

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

  target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
      [](mlir::arith::ConstantOp op) {
        return isa<IndexType>(op.getValue().getType());
      });

  target.addDynamicallyLegalOp<memref::LoadOp>([&](Operation *op) {
    auto users = cast<memref::LoadOp>(op).getResult().getUsers();
    if (cast<memref::LoadOp>(op).getResult().getDefiningOp()) {
      if (isa<memref::GetGlobalOp>(
              cast<memref::LoadOp>(op).getResult().getDefiningOp())) {
        auto detectable = llvm::any_of(users, [](Operation *user) {
          return isa<mod_arith::EncapsulateOp>(user);
        });
        return detectable;
      }
    }

    return (typeConverter.isLegal(op->getOperandTypes()) &&
            typeConverter.isLegal(op->getResultTypes()));
  });

  target.addDynamicallyLegalOp<
      memref::AllocOp, memref::DeallocOp, memref::StoreOp, memref::SubViewOp,
      memref::CopyOp, tensor::FromElementsOp, tensor::ExtractOp,
      affine::AffineStoreOp, affine::AffineLoadOp>([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  RewritePatternSet patterns(context);
  patterns
      .add<ConvertConstant, ConvertExtSI, ConvertExtUI,
           ConvertBinOp<mlir::arith::AddIOp, mod_arith::AddOp>,
           ConvertBinOp<mlir::arith::SubIOp, mod_arith::SubOp>,
           ConvertBinOp<mlir::arith::MulIOp, mod_arith::MulOp>, ConvertLoadOp,
           ConvertAny<memref::AllocOp>, ConvertAny<memref::DeallocOp>,
           ConvertAny<memref::StoreOp>, ConvertAny<memref::SubViewOp>,
           ConvertAny<memref::CopyOp>, ConvertAny<tensor::FromElementsOp>,
           ConvertAny<tensor::ExtractOp>, ConvertAny<affine::AffineStoreOp>,
           ConvertAny<affine::AffineLoadOp>>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace arith
}  // namespace heir
}  // namespace mlir