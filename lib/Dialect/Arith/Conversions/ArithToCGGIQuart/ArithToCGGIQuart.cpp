#include "lib/Dialect/Arith/Conversions/ArithToCGGIQuart/ArithToCGGIQuart.h"

#include <mlir/IR/MLIRContext.h>

#include <cstdint>

#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::arith {

#define GEN_PASS_DEF_ARITHTOCGGIQUART
#include "lib/Dialect/Arith/Conversions/ArithToCGGIQuart/ArithToCGGIQuart.h.inc"

static constexpr unsigned maxIntWidth = 16;

static lwe::LWECiphertextType convertArithToCGGIType(IntegerType type,
                                                     MLIRContext *ctx) {
  return lwe::LWECiphertextType::get(ctx,
                                     lwe::UnspecifiedBitFieldEncodingAttr::get(
                                         ctx, type.getIntOrFloatBitWidth()),
                                     lwe::LWEParamsAttr());
}

static std::optional<Type> convertArithToCGGIQuartType(IntegerType type,
                                                       MLIRContext *ctx) {
  auto lweType = lwe::LWECiphertextType::get(
      ctx, lwe::UnspecifiedBitFieldEncodingAttr::get(ctx, maxIntWidth),
      lwe::LWEParamsAttr());

  float width = type.getWidth();
  float realWidth = maxIntWidth >> 1;

  uint8_t nbChunks = ceil(width / realWidth);

  if (width > 64) return std::nullopt;

  return RankedTensorType::get({nbChunks}, lweType);
}

static std::optional<Type> convertArithLikeToCGGIQuartType(ShapedType type,
                                                           MLIRContext *ctx) {
  if (auto arithType = llvm::dyn_cast<IntegerType>(type.getElementType())) {
    float width = arithType.getWidth();
    float realWidth = maxIntWidth >> 1;

    uint8_t nbChunks = ceil(width / realWidth);

    if (width > 64) return std::nullopt;

    if (arithType.getIntOrFloatBitWidth() == maxIntWidth)
      return convertArithToCGGIQuartType(arithType, ctx);

    auto newShape = to_vector(type.getShape());
    newShape.push_back(nbChunks);

    if (isa<RankedTensorType>(type)) {
      return RankedTensorType::get(
          newShape, IntegerType::get(type.getContext(), maxIntWidth));
    }

    if (isa<MemRefType>(type)) {
      return MemRefType::get(newShape,
                             IntegerType::get(type.getContext(), maxIntWidth));
    }
  }
  return type;
}

class ArithToCGGIQuartTypeConverter : public TypeConverter {
 public:
  ArithToCGGIQuartTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // Convert Integer types to LWE ciphertext types
    addConversion([ctx](IntegerType type) -> std::optional<Type> {
      return convertArithToCGGIQuartType(type, ctx);
    });

    addConversion([ctx](ShapedType type) -> std::optional<Type> {
      return convertArithLikeToCGGIQuartType(type, ctx);
    });
  }
};

static Value createTrivialOpMaxWidth(ImplicitLocOpBuilder b, int value) {
  auto maxWideIntType = IntegerType::get(b.getContext(), maxIntWidth >> 1);
  auto intAttr = b.getIntegerAttr(maxWideIntType, value);

  auto encoding =
      lwe::UnspecifiedBitFieldEncodingAttr::get(b.getContext(), maxIntWidth);
  auto lweType = lwe::LWECiphertextType::get(b.getContext(), encoding,
                                             lwe::LWEParamsAttr());

  return b.create<cggi::CreateTrivialOp>(lweType, intAttr);
}

/// Extracts the `input` tensor slice with elements at the last dimension offset
/// by `lastOffset`. Returns a value of tensor type with the last dimension
/// reduced to x1 or fully scalarized, e.g.:
///   - tensor<2xi16>   --> i16
static Value extractLastDimSlice(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input,
                                 int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<RankedTensorType>(input.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Create index element
  auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
  auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
  SmallVector<Value, 1> indices;
  indices.push_back(constantOp.getResult());

  // Scalarize the result in case of 1D tensors.
  if (shape.size() == 1) {
    return rewriter.create<tensor::ExtractOp>(loc, input, indices);
  }

  SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  offsets.back() = rewriter.getIndexAttr(lastOffset);
  SmallVector<OpFoldResult> sizes(shape.size());
  sizes.back() = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));

  return rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes,
                                                 strides);
}

/// Extracts four tensor slices from the `input` whose type is `tensor<...x4T>`,
/// with the first element at offset 0, second element at offset 1 and so on.
static SmallVector<Value> extractLastDimHalves(
    ConversionPatternRewriter &rewriter, Location loc, Value input) {
  auto tenShape = cast<ShapedType>(input.getType()).getShape();
  auto nbChunks = tenShape.back();
  SmallVector<Value> newTrivialOps;

  for (int i = 0; i < nbChunks; ++i) {
    newTrivialOps.push_back(extractLastDimSlice(rewriter, loc, input, i));
  }

  return newTrivialOps;
};

static Value createScalarOrSplatConstant(OpBuilder &builder, Location loc,
                                         Type type, int64_t value) {
  unsigned elementBitWidth = 0;
  if (auto lweTy = dyn_cast<lwe::LWECiphertextType>(type))
    elementBitWidth =
        cast<lwe::UnspecifiedBitFieldEncodingAttr>(lweTy.getEncoding())
            .getCleartextBitwidth();
  else
    elementBitWidth = maxIntWidth;

  auto apValue = APInt(elementBitWidth, value);

  auto maxWideIntType =
      IntegerType::get(builder.getContext(), maxIntWidth >> 1);
  auto intAttr = builder.getIntegerAttr(maxWideIntType, value);

  return builder.create<cggi::CreateTrivialOp>(loc, type, intAttr);
}

static Value insertLastDimSlice(ConversionPatternRewriter &rewriter,
                                Location loc, Value source, Value dest,
                                int64_t lastOffset) {
  assert(lastOffset <
             cast<RankedTensorType>(dest.getType()).getShape().back() &&
         "Offset out of bounds");

  // // Handle scalar source.
  auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
  auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
  SmallVector<Value, 1> indices;
  indices.push_back(constantOp.getResult());

  return rewriter.create<tensor::InsertOp>(loc, source, dest, indices);
}

/// Constructs a new tensor of type `resultType` by creating a series of
/// insertions of `resultComponents`, each at the next offset of the last tensor
/// dimension.
/// When all `resultComponents` are scalars, the result type is `tensor<NxT>`;
/// when `resultComponents` are `tensor<...x1xT>`s, the result type is
/// `tensor<...xNxT>`, where `N` is the number of `resultComponents`.
static Value constructResultTensor(ConversionPatternRewriter &rewriter,
                                   Location loc, RankedTensorType resultType,
                                   ValueRange resultComponents) {
  Value resultVec = createScalarOrSplatConstant(rewriter, loc, resultType, 0);
  for (auto [i, component] : llvm::enumerate(resultComponents))
    resultVec = insertLastDimSlice(rewriter, loc, component, resultVec, i);

  return resultVec;
}

struct ConvertQuartConstantOp
    : public OpConversionPattern<mlir::arith::ConstantOp> {
  ConvertQuartConstantOp(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::arith::ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isa<IndexType>(op.getValue().getType())) {
      return failure();
    }
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type oldType = op.getType();
    auto newType = getTypeConverter()->convertType<RankedTensorType>(oldType);
    auto acutalBitWidth = maxIntWidth >> 1;

    if (!newType)
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("unsupported type: {0}", op.getType()));

    Attribute oldValue = op.getValueAttr();
    auto tenShape = newType.getShape();
    auto nbChunks = tenShape.back();
    SmallVector<Value, 1> newTrivialOps;

    if (auto intAttr = dyn_cast<IntegerAttr>(oldValue)) {
      for (uint8_t i = 0; i < nbChunks; i++) {
        APInt intChunck =
            intAttr.getValue().extractBits(acutalBitWidth, i * acutalBitWidth);

        auto encrypt = createTrivialOpMaxWidth(b, intChunck.getSExtValue());
        newTrivialOps.push_back(encrypt);
      }

      Value resultVec =
          constructResultTensor(rewriter, op.getLoc(), newType, newTrivialOps);
      rewriter.replaceOp(op, resultVec);

      return success();
    }
    return failure();
  }
};

template <typename ArithExtOp>
struct ConvertQuartExt final : OpConversionPattern<ArithExtOp> {
  using OpConversionPattern<ArithExtOp>::OpConversionPattern;

  // Since each type inside the program is a tensor with 4 elements, we can
  // simply return the input tensor as the result. The generated code will later
  // be removed by the CSE pass.

  LogicalResult matchAndRewrite(
      ArithExtOp op, typename ArithExtOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto newResultTy = cast<ShapedType>(
        convertArithToCGGIQuartType(cast<IntegerType>(op.getResult().getType()),
                                    op.getContext())
            .value());
    auto newInTy = cast<ShapedType>(
        convertArithToCGGIQuartType(cast<IntegerType>(op.getIn().getType()),
                                    op.getContext())
            .value());

    auto resultChunks = newResultTy.getShape().back();
    auto inChunks = newInTy.getShape().back();

    if (resultChunks > inChunks) {
      auto paddingFactor = resultChunks - inChunks;

      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(paddingFactor));

      auto padValue = createTrivialOpMaxWidth(b, 0);

      auto resultVec = b.create<tensor::PadOp>(newResultTy, adaptor.getIn(),
                                               low, high, padValue,
                                               /*nofold=*/true);

      rewriter.replaceOp(op, resultVec);
      return success();
    }
    return failure();
  }
};

struct ConvertQuartAddI final : OpConversionPattern<mlir::arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::AddIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    SmallVector<Value> splitLhs =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    SmallVector<Value> splitRhs =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    assert(splitLhs.size() == splitRhs.size() && "Mismatched tensor sizes");

    // Actual type of the underlying elements; we use half the width.
    // Create Constant
    auto intAttr = IntegerAttr::get(rewriter.getI8Type(), maxIntWidth >> 1);

    auto elemType = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth), op->getContext());
    auto realTy = convertArithToCGGIType(
        IntegerType::get(op->getContext(), maxIntWidth >> 1), op->getContext());

    auto constantOp = b.create<mlir::arith::ConstantOp>(intAttr);

    SmallVector<Value> carries;
    SmallVector<Value> outputs;

    for (int i = 0; i < splitLhs.size(); ++i) {
      auto lowSum = b.create<cggi::AddOp>(splitLhs[i], splitRhs[i]);
      auto outputLsb = b.create<cggi::CastOp>(op.getLoc(), realTy, lowSum);
      auto outputLsbHigh =
          b.create<cggi::CastOp>(op.getLoc(), elemType, outputLsb);

      // Now all the outputs are 16b elements, wants presentation of 4x8b
      if (i != splitLhs.size() - 1) {
        auto carry = b.create<cggi::ShiftRightOp>(elemType, lowSum, constantOp);
        carries.push_back(carry);
      }

      if (i == 0) {
        outputs.push_back(outputLsbHigh);
      } else {
        auto high = b.create<cggi::AddOp>(outputLsbHigh, carries[i - 1]);
        outputs.push_back(high);
      }
    }

    Value resultVec = constructResultTensor(rewriter, loc, newTy, outputs);
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct ArithToCGGIQuart : public impl::ArithToCGGIQuartBase<ArithToCGGIQuart> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ArithToCGGIQuartTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<cggi::CGGIDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };

    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      tensor::TensorDialect>(opLegalCallback);

    target.addDynamicallyLegalOp<
        memref::AllocOp, memref::DeallocOp, memref::StoreOp, memref::LoadOp,
        memref::SubViewOp, memref::CopyOp, affine::AffineLoadOp,
        affine::AffineStoreOp, tensor::FromElementsOp, tensor::ExtractOp>(
        [&](Operation *op) {
          return typeConverter.isLegal(op->getOperandTypes()) &&
                 typeConverter.isLegal(op->getResultTypes());
        });

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [](mlir::arith::ConstantOp op) {
          // Allow use of constant if it is used to denote the size of a shift
          bool usedByShift = llvm::any_of(op->getUsers(), [&](Operation *user) {
            return isa<cggi::ShiftRightOp>(user);
          });
          return (isa<IndexType>(op.getValue().getType()) || (usedByShift));
        });

    patterns.add<
        ConvertQuartConstantOp, ConvertQuartExt<mlir::arith::ExtUIOp>,
        ConvertQuartExt<mlir::arith::ExtSIOp>, ConvertQuartAddI,
        ConvertAny<memref::LoadOp>, ConvertAny<memref::AllocOp>,
        ConvertAny<memref::DeallocOp>, ConvertAny<memref::StoreOp>,
        ConvertAny<memref::SubViewOp>, ConvertAny<memref::CopyOp>,
        ConvertAny<tensor::FromElementsOp>, ConvertAny<tensor::ExtractOp>,
        ConvertAny<affine::AffineStoreOp>, ConvertAny<affine::AffineLoadOp>>(
        typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::arith
