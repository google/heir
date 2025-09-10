#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_DEF_ARITHTOMODARITH
#include "lib/Dialect/Arith/Conversions/ArithToModArith/ArithToModArith.h.inc"

IntegerType getIntegerType(Type type, int64_t modulus) {
  if (modulus == 0) {
    auto modulusBitSize = (int64_t)type.getIntOrFloatBitWidth();
    modulus = (1L << (modulusBitSize - 1L));
    return mlir::IntegerType::get(type.getContext(), modulusBitSize + 1);
  }
  return mlir::IntegerType::get(type.getContext(), 64);
}

static mod_arith::ModArithType convertArithType(Type type, int64_t modulus) {
  auto integerType = getIntegerType(type, modulus);
  if (modulus == 0) {
    auto modulusBitSize = (int64_t)type.getIntOrFloatBitWidth();
    modulus = (1L << (modulusBitSize - 1L));
  }
  return mod_arith::ModArithType::get(
      type.getContext(), mlir::IntegerAttr::get(integerType, modulus));
}

static Type convertArithLikeType(ShapedType type, int64_t modulus) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    return type.cloneWith(type.getShape(),
                          convertArithType(arithType, modulus));
  }
  return type;
}

static auto buildLoadOps(int64_t modulus) {
  return [=](OpBuilder& builder, Type resultTypes, ValueRange inputs,
             Location loc) -> Value {
    auto b = ImplicitLocOpBuilder(loc, builder);

    assert(inputs.size() == 1);
    auto loadOp = inputs[0].getDefiningOp<memref::LoadOp>();

    if (!loadOp) return {};

    auto* globaMemReflOp = loadOp.getMemRef().getDefiningOp();

    if (!globaMemReflOp) return {};

    auto integerType = getIntegerType(loadOp.getType(), modulus);
    auto modArithType = convertArithType(loadOp.getType(), modulus);
    auto extui =
        mlir::arith::ExtUIOp::create(b, integerType, loadOp.getResult());

    return mod_arith::EncapsulateOp::create(b, modArithType, extui);
  };
}

class ArithToModArithTypeConverter : public TypeConverter {
 public:
  ArithToModArithTypeConverter(MLIRContext* ctx, int64_t modulus) {
    addConversion([](Type type) { return type; });
    addConversion([=](IntegerType type) -> mod_arith::ModArithType {
      return convertArithType(type, modulus);
    });
    addConversion([=](ShapedType type) -> Type {
      return convertArithLikeType(type, modulus);
    });
    addTargetMaterialization(buildLoadOps(modulus));
  }
};

struct ConvertConstant : public OpConversionPattern<mlir::arith::ConstantOp> {
  ConvertConstant(mlir::MLIRContext* context)
      : OpConversionPattern<mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (isa<IndexType>(op.getValue().getType())) {
      return failure();
    }

    Type newResultType = typeConverter->convertType(op.getResult().getType());
    mod_arith::ModArithType newResultEltTy =
        cast<mod_arith::ModArithType>(getElementTypeOrSelf(newResultType));
    IntegerType storageType =
        cast<IntegerType>(newResultEltTy.getModulus().getType());
    unsigned newWidth = storageType.getWidth();

    if (auto denseIntAttr = dyn_cast<DenseIntElementsAttr>(op.getValue())) {
      auto shapedType = cast<ShapedType>(newResultType);
      SmallVector<APInt, 4> newValues;
      for (const APInt& val : denseIntAttr.getValues<APInt>()) {
        newValues.push_back(val.sextOrTrunc(newWidth));
      }
      TypedAttr newAttr = DenseIntElementsAttr::get(
          RankedTensorType::get(shapedType.getShape(), storageType), newValues);
      auto result = mod_arith::ConstantOp::create(b, shapedType, newAttr);
      rewriter.replaceOp(op, result);
      return success();
    }

    IntegerAttr oldAttr = cast<IntegerAttr>(op.getValue());
    IntegerAttr newAttr =
        IntegerAttr::get(storageType, oldAttr.getValue().zextOrTrunc(newWidth));
    auto result = mod_arith::ConstantOp::create(b, newResultType, newAttr);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertExtSI : public OpConversionPattern<mlir::arith::ExtSIOp> {
  ConvertExtSI(mlir::MLIRContext* context)
      : OpConversionPattern<mlir::arith::ExtSIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ExtSIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = mod_arith::ModSwitchOp::create(
        b, op.getLoc(), typeConverter->convertType(op.getType()),
        adaptor.getIn());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertExtUI : public OpConversionPattern<mlir::arith::ExtUIOp> {
  ConvertExtUI(mlir::MLIRContext* context)
      : OpConversionPattern<mlir::arith::ExtUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = mod_arith::ModSwitchOp::create(
        b, op.getLoc(), typeConverter->convertType(op.getType()),
        adaptor.getIn());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLoadOp : public OpConversionPattern<mlir::memref::LoadOp> {
  ConvertLoadOp(mlir::MLIRContext* context)
      : OpConversionPattern<mlir::memref::LoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ::mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto defineOp = op.getMemRef().getDefiningOp();

    if (op.getMemRef().getDefiningOp()) {
      if (isa<memref::GetGlobalOp>(defineOp)) {
        // skip global memref
        return success();
      }
    }

    auto result = memref::LoadOp::create(
        rewriter, op.getLoc(), typeConverter->convertType(op.getType()),
        adaptor.getOperands()[0], op.getIndices());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithToModArith : impl::ArithToModArithBase<ArithToModArith> {
  using ArithToModArithBase::ArithToModArithBase;

  void runOnOperation() override;
};

void ArithToModArith::runOnOperation() {
  MLIRContext* context = &getContext();
  ModuleOp module = getOperation();
  ArithToModArithTypeConverter typeConverter(context, modulus);

  ConversionTarget target(*context);
  target.addLegalDialect<mod_arith::ModArithDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect>();

  target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
      [](mlir::arith::ConstantOp op) {
        return isa<IndexType>(op.getValue().getType());
      });
  target.addDynamicallyLegalOp<mlir::arith::AddIOp, mlir::arith::SubIOp,
                               mlir::arith::MulIOp, mlir::arith::RemUIOp>(
      [](Operation* op) {
        return isa<IndexType>(op->getOperand(0).getType());
      });

  target.addDynamicallyLegalOp<memref::LoadOp>([&](Operation* op) {
    auto users = cast<memref::LoadOp>(op).getResult().getUsers();
    if (cast<memref::LoadOp>(op).getResult().getDefiningOp()) {
      if (isa<memref::GetGlobalOp>(
              cast<memref::LoadOp>(op).getResult().getDefiningOp())) {
        auto detectable = llvm::any_of(users, [](Operation* user) {
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
      tensor::ExtractSliceOp, tensor::InsertOp, tensor::ExpandShapeOp,
      tensor::ConcatOp, affine::AffineStoreOp, affine::AffineLoadOp,
      affine::AffineForOp, affine::AffineYieldOp, tensor_ext::RotateOp>(
      [&](Operation* op) {
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
           ConvertAny<tensor::ExtractOp>, ConvertAny<tensor::ExtractSliceOp>,
           ConvertAny<tensor::InsertOp>, ConvertAny<tensor::ExpandShapeOp>,
           ConvertAny<tensor::ConcatOp>, ConvertAny<affine::AffineStoreOp>,
           ConvertAny<affine::AffineLoadOp>, ConvertAny<affine::AffineForOp>,
           ConvertAny<affine::AffineYieldOp>, ConvertAny<tensor_ext::RotateOp>>(
          typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(module, target, std::move(patterns),
                                    config))) {
    signalPassFailure();
  }
}

}  // namespace arith
}  // namespace heir
}  // namespace mlir
