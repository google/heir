#include "lib/Dialect/Arith/Conversions/ArithToCGGI/ArithToCGGI.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
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
      .Case<mlir::arith::SubIOp, mlir::arith::AddIOp, mlir::arith::MulIOp>(
          [](auto op) {
            // This lambda will be called for any of the matched operation types
            if (auto lhsDefOp = op.getOperand(0).getDefiningOp()) {
              auto lshAllowed = allowedRemainArith(lhsDefOp);
              if (auto rhsDefOp = op.getOperand(1).getDefiningOp()) {
                auto rhsAllowed = allowedRemainArith(rhsDefOp);
                return lshAllowed && rhsAllowed;
              }
            }
            return false;
          })
      .Default([](Operation *) {
        // Default case for operations that don't match any of the types
        return false;
      });
}

static bool hasLWEAnnotation(Operation *op) {
  mlir::StringAttr check =
      op->getAttrOfType<mlir::StringAttr>("lwe_annotation");

  if (check) return true;

  // Check recursively if a defining op has a LWE annotation
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<mlir::arith::ExtUIOp, mlir::arith::ExtSIOp, mlir::arith::TruncIOp>(
          [](auto op) {
            if (auto *defOp = op.getIn().getDefiningOp()) {
              return hasLWEAnnotation(defOp);
            }
            return op->template getAttrOfType<mlir::StringAttr>(
                       "lwe_annotation") != nullptr;
          })
      .Case<mlir::arith::SubIOp, mlir::arith::AddIOp, mlir::arith::MulIOp>(
          [](auto op) {
            // This lambda will be called for any of the matched operation types
            if (auto lhsDefOp = op.getOperand(0).getDefiningOp()) {
              auto lshAllowed = hasLWEAnnotation(lhsDefOp);
              if (auto rhsDefOp = op.getOperand(1).getDefiningOp()) {
                auto rhsAllowed = hasLWEAnnotation(rhsDefOp);
                return lshAllowed || rhsAllowed;
              }
            }
            return false;
          })
      .Default([](Operation *) { return false; });
}

static Value materializeTarget(OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (!isa<IntegerType>(inputType))
    llvm_unreachable(
        "Non-integer types should never be the input to a materializeTarget.");

  if (auto inValue = inputs.front().getDefiningOp<mlir::arith::ConstantOp>()) {
    auto intAttr = cast<IntegerAttr>(inValue.getValueAttr());

    return builder.create<cggi::CreateTrivialOp>(loc, type, intAttr);
  }
  // Comes from function/loop argument: Trivial encrypt through LWE
  auto encoding = cast<lwe::LWECiphertextType>(type).getEncoding();
  auto ptxtTy = lwe::LWEPlaintextType::get(builder.getContext(), encoding);
  return builder.create<lwe::TrivialEncryptOp>(
      loc, type,
      builder.create<lwe::EncodeOp>(loc, ptxtTy, inputs[0], encoding),
      lwe::LWEParamsAttr());
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

struct ConvertCmpOp : public OpConversionPattern<mlir::arith::CmpIOp> {
  ConvertCmpOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::CmpIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::CmpIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto lweBooleanType = lwe::LWECiphertextType::get(
        op->getContext(),
        lwe::UnspecifiedBitFieldEncodingAttr::get(op->getContext(), 1),
        lwe::LWEParamsAttr());

    if (auto lhsDefOp = op.getLhs().getDefiningOp()) {
      if (!hasLWEAnnotation(lhsDefOp) && allowedRemainArith(lhsDefOp)) {
        auto result = b.create<cggi::CmpOp>(lweBooleanType, op.getPredicate(),
                                            adaptor.getRhs(), op.getLhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    if (auto rhsDefOp = op.getRhs().getDefiningOp()) {
      if (!hasLWEAnnotation(rhsDefOp) && allowedRemainArith(rhsDefOp)) {
        auto result = b.create<cggi::CmpOp>(lweBooleanType, op.getPredicate(),
                                            adaptor.getLhs(), op.getRhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    auto cmpOp = b.create<cggi::CmpOp>(lweBooleanType, op.getPredicate(),
                                       adaptor.getLhs(), adaptor.getRhs());

    rewriter.replaceOp(op, cmpOp);
    return success();
  }
};

struct ConvertSubOp : public OpConversionPattern<mlir::arith::SubIOp> {
  ConvertSubOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::SubIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::SubIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    if (auto rhsDefOp = op.getRhs().getDefiningOp()) {
      if (!hasLWEAnnotation(rhsDefOp) && allowedRemainArith(rhsDefOp)) {
        auto result = b.create<cggi::SubOp>(adaptor.getLhs().getType(),
                                            adaptor.getLhs(), op.getRhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    auto subOp = b.create<cggi::SubOp>(adaptor.getLhs().getType(),
                                       adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, subOp);
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<mlir::arith::SelectOp> {
  ConvertSelectOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::SelectOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::SelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmuxOp = b.create<cggi::SelectOp>(
        adaptor.getTrueValue().getType(), adaptor.getCondition(),
        adaptor.getTrueValue(), adaptor.getFalseValue());

    rewriter.replaceOp(op, cmuxOp);
    return success();
  }
};

template <typename SourceArithShOp, typename TargetCGGIShOp>
struct ConvertShOp : public OpConversionPattern<SourceArithShOp> {
  ConvertShOp(mlir::MLIRContext *context)
      : OpConversionPattern<SourceArithShOp>(context) {}

  using OpConversionPattern<SourceArithShOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceArithShOp op, typename SourceArithShOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cteShiftSizeOp =
        op.getRhs().template getDefiningOp<mlir::arith::ConstantOp>();

    if (cteShiftSizeOp) {
      auto outputType = adaptor.getLhs().getType();

      auto shiftAmount = cast<IntegerAttr>(cteShiftSizeOp.getValue())
                             .getValue()
                             .getSExtValue();

      auto inputValue =
          mlir::IntegerAttr::get(rewriter.getIndexType(), (int8_t)shiftAmount);

      auto shiftOp =
          b.create<TargetCGGIShOp>(outputType, adaptor.getLhs(), inputValue);
      rewriter.replaceOp(op, shiftOp);

      return success();
    }

    cteShiftSizeOp =
        op.getLhs().template getDefiningOp<mlir::arith::ConstantOp>();

    auto outputType = adaptor.getRhs().getType();

    auto shiftAmount =
        cast<IntegerAttr>(cteShiftSizeOp.getValue()).getValue().getSExtValue();

    auto inputValue =
        mlir::IntegerAttr::get(rewriter.getIndexType(), shiftAmount);

    auto shiftOp =
        b.create<TargetCGGIShOp>(outputType, adaptor.getLhs(), inputValue);
    rewriter.replaceOp(op, shiftOp);

    return success();
  }
};

template <typename SourceArithOp, typename TargetCGGIOp>
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
        auto result = b.create<TargetCGGIOp>(adaptor.getRhs().getType(),
                                             adaptor.getRhs(), op.getLhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    if (auto rhsDefOp = op.getRhs().getDefiningOp()) {
      if (!hasLWEAnnotation(rhsDefOp) && allowedRemainArith(rhsDefOp)) {
        auto result = b.create<TargetCGGIOp>(adaptor.getLhs().getType(),
                                             adaptor.getLhs(), op.getRhs());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    auto result = b.create<TargetCGGIOp>(adaptor.getLhs().getType(),
                                         adaptor.getLhs(), adaptor.getRhs());
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

    target.addDynamicallyLegalOp<mlir::arith::SubIOp, mlir::arith::AddIOp,
                                 mlir::arith::MulIOp>([&](Operation *op) {
      if (auto *defLhsOp = op->getOperand(0).getDefiningOp()) {
        if (auto *defRhsOp = op->getOperand(1).getDefiningOp()) {
          return !hasLWEAnnotation(defLhsOp) && !hasLWEAnnotation(defRhsOp) &&
                 allowedRemainArith(defLhsOp) && allowedRemainArith(defRhsOp);
        }
      }
      return false;
    });

    target.addDynamicallyLegalOp<mlir::arith::ExtSIOp>([&](Operation *op) {
      if (auto *defOp =
              cast<mlir::arith::ExtSIOp>(op).getOperand().getDefiningOp()) {
        return !hasLWEAnnotation(defOp) && allowedRemainArith(defOp);
      }
      return false;
    });

    target.addDynamicallyLegalOp<mlir::arith::ExtUIOp>([&](Operation *op) {
      if (auto *defOp =
              cast<mlir::arith::ExtUIOp>(op).getOperand().getDefiningOp()) {
        return !hasLWEAnnotation(defOp) && allowedRemainArith(defOp);
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
          return !hasLWEAnnotation(defOp.getValue().getDefiningOp()) &&
                 allowedRemainArith(defOp.getValue().getDefiningOp());
        }
        return false;
      });
      auto allStoreOpsAreArith =
          llvm::all_of(op->getUses(), [&](OpOperand &op) {
            if (auto defOp = dyn_cast<memref::StoreOp>(op.getOwner())) {
              return !hasLWEAnnotation(defOp.getValue().getDefiningOp()) &&
                     allowedRemainArith(defOp.getValue().getDefiningOp());
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
        ConvertTruncIOp, ConvertExtUIOp, ConvertExtSIOp, ConvertSelectOp,
        ConvertCmpOp, ConvertSubOp,
        ConvertShOp<mlir::arith::ShRSIOp, cggi::ScalarShiftRightOp>,
        ConvertShOp<mlir::arith::ShRUIOp, cggi::ScalarShiftRightOp>,
        ConvertShOp<mlir::arith::ShLIOp, cggi::ScalarShiftLeftOp>,
        ConvertArithBinOp<mlir::arith::AddIOp, cggi::AddOp>,
        ConvertArithBinOp<mlir::arith::MulIOp, cggi::MulOp>,
        ConvertArithBinOp<mlir::arith::MaxSIOp, cggi::MaxOp>,
        ConvertArithBinOp<mlir::arith::MinSIOp, cggi::MinOp>,
        ConvertArithBinOp<mlir::arith::MaxUIOp, cggi::MaxOp>,
        ConvertArithBinOp<mlir::arith::MinUIOp, cggi::MinOp>,
        ConvertArithBinOp<mlir::arith::XOrIOp, cggi::XorOp>,
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
