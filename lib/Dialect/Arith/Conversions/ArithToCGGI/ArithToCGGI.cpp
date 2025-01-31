#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::arith {

#define GEN_PASS_DEF_ARITHTOCGGI
#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h.inc"

static lwe::LWECiphertextType convertArithToCGGIType(IntegerType type,
                                                     MLIRContext *ctx) {
  return lwe::LWECiphertextType::get(ctx,
                                     lwe::UnspecifiedBitFieldEncodingAttr::get(
                                         ctx, type.getIntOrFloatBitWidth()),
                                     lwe::LWEParamsAttr());
}

static Type convertArithLikeToCGGIType(ShapedType type, MLIRContext *ctx) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    return type.cloneWith(type.getShape(),
                          convertArithToCGGIType(arithType, ctx));
  }
  return type;
}

// Function to check if an operation is allowed to remain in the Arith dialect
static bool allowedRemainArith(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<mlir::arith::ConstantOp>([](auto op) {
        // This lambda will be called for any of the matched operation types
        return true;
      })
      // Allow memref LoadOp if it comes from a FuncArg or if it comes from
      // an allowed alloc memref
      // Other cases: Memref comes from function -> need to convert to LWE
      .Case<memref::LoadOp>([](memref::LoadOp memrefLoad) {
        return memrefLoad.getMemRef().getDefiningOp() != nullptr;
      })
      .Case<mlir::arith::ExtUIOp, mlir::arith::ExtSIOp, mlir::arith::TruncIOp>(
          [](auto op) {
            // This lambda will be called for any of the matched operation types
            if (auto *defOp = op.getIn().getDefiningOp()) {
              return allowedRemainArith(defOp);
            }
            return false;
          })
      .Default([](Operation *) {
        // Default case for operations that don't match any of the types
        return false;
      });
}

static bool hasLWEAnnotation(Operation *op) {
  return static_cast<bool>(
      op->getAttrOfType<mlir::StringAttr>("lwe_annotation"));
}

static Value materializeTarget(OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (!isa<IntegerType>(inputType))
    llvm_unreachable(
        "Non-integer types should never be the input to a materializeTarget.");

  auto inValue = inputs.front().getDefiningOp<mlir::arith::ConstantOp>();
  auto intAttr = cast<IntegerAttr>(inValue.getValueAttr());

  return builder.create<cggi::CreateTrivialOp>(loc, type, intAttr);
}

class ArithToCGGITypeConverter : public TypeConverter {
 public:
  ArithToCGGITypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // Convert Integer types to LWE ciphertext types
    addConversion([ctx](IntegerType type) -> Type {
      return convertArithToCGGIType(type, ctx);
    });

    addConversion([ctx](ShapedType type) -> Type {
      return convertArithLikeToCGGIType(type, ctx);
    });

    // Target materialization to convert integer constants to LWE ciphertexts
    // by creating a trivial LWE ciphertext
    addTargetMaterialization(materializeTarget);
  }
};

struct ConvertTruncIOp : public OpConversionPattern<mlir::arith::TruncIOp> {
  ConvertTruncIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::TruncIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::TruncIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto outType = convertArithToCGGIType(
        cast<IntegerType>(op.getResult().getType()), op->getContext());
    auto castOp = b.create<cggi::CastOp>(op.getLoc(), outType, adaptor.getIn());

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct ConvertExtUIOp : public OpConversionPattern<mlir::arith::ExtUIOp> {
  ConvertExtUIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto outType = convertArithToCGGIType(
        cast<IntegerType>(op.getResult().getType()), op->getContext());
    auto castOp = b.create<cggi::CastOp>(op.getLoc(), outType, adaptor.getIn());

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct ConvertExtSIOp : public OpConversionPattern<mlir::arith::ExtSIOp> {
  ConvertExtSIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ExtSIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ExtSIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto outType = convertArithToCGGIType(
        cast<IntegerType>(op.getResult().getType()), op->getContext());
    auto castOp = b.create<cggi::CastOp>(op.getLoc(), outType, adaptor.getIn());

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct ConvertShRUIOp : public OpConversionPattern<mlir::arith::ShRUIOp> {
  ConvertShRUIOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ShRUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ShRUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cteShiftSizeOp = op.getRhs().getDefiningOp<mlir::arith::ConstantOp>();

    if (cteShiftSizeOp) {
      auto outputType = adaptor.getLhs().getType();

      auto shiftAmount = cast<IntegerAttr>(cteShiftSizeOp.getValue())
                             .getValue()
                             .getSExtValue();

      auto inputValue =
          mlir::IntegerAttr::get(rewriter.getIndexType(), (int8_t)shiftAmount);

      auto shiftOp = b.create<cggi::ScalarShiftRightOp>(
          outputType, adaptor.getLhs(), inputValue);
      rewriter.replaceOp(op, shiftOp);

      return success();
    }

    cteShiftSizeOp = op.getLhs().getDefiningOp<mlir::arith::ConstantOp>();

    auto outputType = adaptor.getRhs().getType();

    auto shiftAmount =
        cast<IntegerAttr>(cteShiftSizeOp.getValue()).getValue().getSExtValue();

    auto inputValue =
        mlir::IntegerAttr::get(rewriter.getIndexType(), shiftAmount);

    auto shiftOp = b.create<cggi::ScalarShiftRightOp>(
        outputType, adaptor.getLhs(), inputValue);
    rewriter.replaceOp(op, shiftOp);

    return success();
  }
};

template <typename SourceArithOp, typename TargetModArithOp>
struct ConvertArithBinOp : public OpConversionPattern<SourceArithOp> {
  ConvertArithBinOp(mlir::MLIRContext *context)
      : OpConversionPattern<SourceArithOp>(context) {}

  using OpConversionPattern<SourceArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceArithOp op, typename SourceArithOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (auto lhsDefOp = op.getLhs().getDefiningOp()) {
      if (!hasLWEAnnotation(lhsDefOp) && allowedRemainArith(lhsDefOp)) {
        auto result = b.create<TargetModArithOp>(adaptor.getRhs().getType(),
                                                 adaptor.getRhs(), op.getLhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    if (auto rhsDefOp = op.getRhs().getDefiningOp()) {
      if (!hasLWEAnnotation(rhsDefOp) && allowedRemainArith(rhsDefOp)) {
        auto result = b.create<TargetModArithOp>(adaptor.getLhs().getType(),
                                                 adaptor.getLhs(), op.getRhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    auto result = b.create<TargetModArithOp>(
        adaptor.getLhs().getType(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertAllocOp : public OpConversionPattern<mlir::memref::AllocOp> {
  ConvertAllocOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::memref::AllocOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::memref::AllocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    for (auto *userOp : op->getUsers()) {
      userOp->setAttr("lwe_annotation",
                      mlir::StringAttr::get(userOp->getContext(), "LWE"));
    }

    auto lweType = getTypeConverter()->convertType(op.getType());
    auto allocOp =
        b.create<memref::AllocOp>(op.getLoc(), lweType, op.getOperands());
    rewriter.replaceOp(op, allocOp);
    return success();
  }
};

struct ArithToCGGI : public impl::ArithToCGGIBase<ArithToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ArithToCGGITypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<cggi::CGGIDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addLegalOp<mlir::arith::ConstantOp>();

    target.addDynamicallyLegalOp<mlir::arith::ExtSIOp>([&](Operation *op) {
      if (auto *defOp =
              cast<mlir::arith::ExtSIOp>(op).getOperand().getDefiningOp()) {
        return hasLWEAnnotation(defOp) || allowedRemainArith(defOp);
      }
      return false;
    });

    target.addDynamicallyLegalOp<memref::SubViewOp, memref::CopyOp,
                                 tensor::FromElementsOp, tensor::ExtractOp,
                                 affine::AffineStoreOp, affine::AffineLoadOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });

    target.addDynamicallyLegalOp<memref::AllocOp>([&](Operation *op) {
      // Check if all Store ops are constants or GetGlobals, if not store op,
      // accepts Check if there is at least one Store op that is a constants
      auto containsAnyStoreOp = llvm::any_of(op->getUses(), [&](OpOperand &op) {
        if (auto defOp = dyn_cast<memref::StoreOp>(op.getOwner())) {
          return allowedRemainArith(defOp.getValue().getDefiningOp());
        }
        return false;
      });
      auto allStoreOpsAreArith =
          llvm::all_of(op->getUses(), [&](OpOperand &op) {
            if (auto defOp = dyn_cast<memref::StoreOp>(op.getOwner())) {
              return allowedRemainArith(defOp.getValue().getDefiningOp());
            }
            return true;
          });

      return (allStoreOpsAreArith && containsAnyStoreOp) ||
             // The other case: Memref need to be in LWE format
             (typeConverter.isLegal(op->getOperandTypes()) &&
              typeConverter.isLegal(op->getResultTypes()));
    });

    target.addDynamicallyLegalOp<memref::StoreOp>([&](Operation *op) {
      if (typeConverter.isLegal(op->getOperandTypes()) &&
          typeConverter.isLegal(op->getResultTypes())) {
        return true;
      }

      if (auto lweAttr =
              op->getAttrOfType<mlir::StringAttr>("lwe_annotation")) {
        return false;
      }

      if (auto *defOp = cast<memref::StoreOp>(op).getValue().getDefiningOp()) {
        if (isa<mlir::arith::ConstantOp>(defOp) ||
            isa<mlir::memref::GetGlobalOp>(defOp)) {
          return true;
        }
      }
      return true;
    });

    // Convert LoadOp if memref comes from an argument
    target.addDynamicallyLegalOp<memref::LoadOp>([&](Operation *op) {
      if (typeConverter.isLegal(op->getOperandTypes()) &&
          typeConverter.isLegal(op->getResultTypes())) {
        return true;
      }

      if (dyn_cast<memref::LoadOp>(op).getMemRef().getDefiningOp() == nullptr) {
        return false;
      }

      if (auto lweAttr =
              op->getAttrOfType<mlir::StringAttr>("lwe_annotation")) {
        return false;
      }

      return true;
    });

    // Convert Dealloc if memref comes from an argument
    target.addDynamicallyLegalOp<memref::DeallocOp>([&](Operation *op) {
      if (typeConverter.isLegal(op->getOperandTypes()) &&
          typeConverter.isLegal(op->getResultTypes())) {
        return true;
      }

      if (auto lweAttr =
              op->getAttrOfType<mlir::StringAttr>("lwe_annotation")) {
        return false;
      }

      return true;
    });

    patterns.add<
        ConvertTruncIOp, ConvertExtUIOp, ConvertExtSIOp, ConvertShRUIOp,
        ConvertArithBinOp<mlir::arith::AddIOp, cggi::AddOp>,
        ConvertArithBinOp<mlir::arith::MulIOp, cggi::MulOp>,
        ConvertArithBinOp<mlir::arith::SubIOp, cggi::SubOp>,
        ConvertAny<memref::LoadOp>, ConvertAllocOp,
        ConvertAny<memref::DeallocOp>, ConvertAny<memref::SubViewOp>,
        ConvertAny<memref::CopyOp>, ConvertAny<memref::StoreOp>,
        ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::ExtractOp>,
        ConvertAny<affine::AffineStoreOp>, ConvertAny<affine::AffineLoadOp> >(
        typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::arith
