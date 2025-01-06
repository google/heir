#include "lib/Dialect/Arith/Transforms/QuarterWideInt.h"

#include <optional>

#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/APInt.h"                // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith {

#define GEN_PASS_DEF_QUARTERWIDEINT
#include "lib/Dialect/Arith/Transforms/Passes.h.inc"

static constexpr unsigned maxIntWidth = 16;

class QuarterWideTypeConverter : public TypeConverter {
 public:
  QuarterWideTypeConverter(MLIRContext *ctx) {
    // Allow unknown types.
    addConversion([](Type ty) -> std::optional<Type> { return ty; });

    // Scalar case.
    addConversion([](IntegerType ty) -> std::optional<Type> {
      unsigned width = ty.getWidth();
      if (width <= maxIntWidth) return ty;

      // i2N --> tensor<4xiN>
      if (width == 2 * maxIntWidth)
        return RankedTensorType::get(
            4, IntegerType::get(ty.getContext(), maxIntWidth));

      return std::nullopt;
    });

    // tensor case.
    addConversion([](ShapedType ty) -> std::optional<Type> {
      auto intTy = dyn_cast<IntegerType>(ty.getElementType());
      if (!intTy) return ty;

      unsigned width = intTy.getWidth();
      if (width <= maxIntWidth) return ty;

      // tensor<...xi2N> --> tensor<...x4xiN>
      if (width == 2 * maxIntWidth) {
        auto newShape = to_vector(ty.getShape());
        newShape.push_back(4);
        return RankedTensorType::get(
            newShape, IntegerType::get(ty.getContext(), maxIntWidth));
      }
      return std::nullopt;
    });
  }
};

//===----------------------------------------------------------------------===//
// Common Helper Functions
//===----------------------------------------------------------------------===//

/// Returns the number divided into four chunks of N/2 bits from `value`, where
/// N = `newBitWidth/2`. Treats `value` as a 2*N bits-wide integer. The bottom
/// bits are returned in the first pair element, while the top bits in the
/// fourth one.
std::tuple<APInt, APInt, APInt, APInt> getQuarters(const APInt &value,
                                                   unsigned newBitWidth) {
  auto acutalBitWidth = newBitWidth >> 1;

  APInt low = value.extractBits(acutalBitWidth, 0);
  APInt midLow = value.extractBits(acutalBitWidth, acutalBitWidth);
  APInt midHigh = value.extractBits(acutalBitWidth, 2 * acutalBitWidth);
  APInt high = value.extractBits(acutalBitWidth, 3 * acutalBitWidth);
  return {std::move(low), std::move(midLow), std::move(midHigh),
          std::move(high)};
}

/// Returns the type with the last (innermost) dimension reduced to x1.
/// Scalarizes 1D tensor inputs to match how we extract/insert tensor values,
/// e.g.:
///   - tensor<3x2xi16> --> tensor<3x1xi16>
///   - tensor<2xi16>   --> i16
Type reduceInnermostDim(RankedTensorType type) {
  if (type.getShape().size() == 1) return type.getElementType();

  auto newShape = to_vector(type.getShape());
  newShape.back() = 1;
  return RankedTensorType::get(newShape, type.getElementType());
}

/// Extracts the `input` tensor slice with elements at the last dimension offset
/// by `lastOffset`. Returns a value of tensor type with the last dimension
/// reduced to x1 or fully scalarized, e.g.:
///   - tensor<2xi16>   --> i16
Value extractLastDimSlice(ConversionPatternRewriter &rewriter, Location loc,
                          Value input, int64_t lastOffset) {
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
std::tuple<Value, Value, Value, Value> extractLastDimHalves(
    ConversionPatternRewriter &rewriter, Location loc, Value input) {
  return {extractLastDimSlice(rewriter, loc, input, 0),
          extractLastDimSlice(rewriter, loc, input, 1),
          extractLastDimSlice(rewriter, loc, input, 2),
          extractLastDimSlice(rewriter, loc, input, 3)};
}

/// Inserts the `source` tensor slice into the `dest` tensor at offset
/// `lastOffset` in the last dimension. `source` can be a scalar when `dest` is
/// a 1D tensor.
Value insertLastDimSlice(ConversionPatternRewriter &rewriter, Location loc,
                         Value source, Value dest, int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<RankedTensorType>(dest.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Handle scalar source.
  if (isa<IntegerType>(source.getType())) {
    auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
    auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
    SmallVector<Value, 1> indices;
    indices.push_back(constantOp.getResult());

    return rewriter.create<tensor::InsertOp>(loc, source, dest, indices);
  }

  SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  offsets.back() = rewriter.getIndexAttr(lastOffset);
  SmallVector<OpFoldResult> sizes(shape.size());
  sizes.back() = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));

  return rewriter.create<tensor::InsertSliceOp>(loc, source, dest, offsets,
                                                sizes, strides);
}

Value createScalarOrSplatConstant(OpBuilder &builder, Location loc, Type type,
                                  int64_t value) {
  unsigned elementBitWidth = 0;
  if (auto intTy = dyn_cast<IntegerType>(type))
    elementBitWidth = intTy.getWidth();
  else
    elementBitWidth = cast<ShapedType>(type).getElementTypeBitWidth();

  auto apValue = APInt(elementBitWidth, value);

  TypedAttr attr;
  if (isa<IntegerType>(type)) {
    attr = builder.getIntegerAttr(type, apValue);
  } else {
    auto vecTy = cast<ShapedType>(type);
    attr = SplatElementsAttr::get(vecTy, apValue);
  }

  return builder.create<mlir::arith::ConstantOp>(loc, attr);
}

/// Constructs a new tensor of type `resultType` by creating a series of
/// insertions of `resultComponents`, each at the next offset of the last tensor
/// dimension.
/// When all `resultComponents` are scalars, the result type is `tensor<NxT>`;
/// when `resultComponents` are `tensor<...x1xT>`s, the result type is
/// `tensor<...xNxT>`, where `N` is the number of `resultComponents`.
Value constructResultTensor(ConversionPatternRewriter &rewriter, Location loc,
                            RankedTensorType resultType,
                            ValueRange resultComponents) {
  Value resultVec = createScalarOrSplatConstant(rewriter, loc, resultType, 0);
  for (auto [i, component] : llvm::enumerate(resultComponents))
    resultVec = insertLastDimSlice(rewriter, loc, component, resultVec, i);

  return resultVec;
}

struct ConvertAddI final : OpConversionPattern<mlir::arith::AddIOp> {
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

    Type elemTy = reduceInnermostDim(newTy);

    auto [lhsElem0, lhsElem1, lhsElem2, lhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1, rhsElem2, rhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // Actual type of the underlying elements; we use half the width.
    auto realTy = IntegerType::get(op.getContext(), maxIntWidth >> 1);
    // Create Constant
    auto intAttr = rewriter.getIntegerAttr(elemTy, maxIntWidth >> 1);
    auto constantOp = b.create<mlir::arith::ConstantOp>(intAttr);

    auto lowSum0 = b.create<mlir::arith::AddIOp>(lhsElem0, rhsElem0);
    auto lowSum1 = b.create<mlir::arith::AddIOp>(lhsElem1, rhsElem1);
    auto lowSum2 = b.create<mlir::arith::AddIOp>(lhsElem2, rhsElem2);
    auto lowSum3 = b.create<mlir::arith::AddIOp>(lhsElem3, rhsElem3);

    auto output0Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum0);
    auto output0LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output0Lsb);

    auto output1Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum1);
    auto output1LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output1Lsb);

    auto output2Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum2);
    auto output2LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output2Lsb);

    auto output3Lsb = b.create<mlir::arith::TruncIOp>(realTy, lowSum3);
    auto output3LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output3Lsb);

    // Now all the outputs are 16b elements, wants presentation of 4x8b
    auto carry0 =
        b.create<mlir::arith::ShRUIOp>(lowSum0, constantOp.getResult());
    auto carry1 =
        b.create<mlir::arith::ShRUIOp>(lowSum1, constantOp.getResult());
    auto carry2 =
        b.create<mlir::arith::ShRUIOp>(lowSum2, constantOp.getResult());

    auto high1 = b.create<mlir::arith::AddIOp>(output1LsbHigh, carry0);
    auto high2 = b.create<mlir::arith::AddIOp>(output2LsbHigh, carry1);
    auto high3 = b.create<mlir::arith::AddIOp>(output3LsbHigh, carry2);

    Value resultVec = constructResultTensor(
        rewriter, loc, newTy, {output0LsbHigh, high1, high2, high3});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

// Implemented using the Karatsuba algorithm
// https://en.wikipedia.org/wiki/Karatsuba_algorithm#Algorithm
struct ConvertMulI final : OpConversionPattern<mlir::arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::MulIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    auto elemTy = reduceInnermostDim(newTy);
    // Actual type of the underlying elements; we use half the width.
    auto realTy = IntegerType::get(op.getContext(), maxIntWidth >> 1);

    // Create Constant
    auto intAttr = rewriter.getIntegerAttr(elemTy, maxIntWidth >> 1);
    auto constantOp = b.create<mlir::arith::ConstantOp>(intAttr);

    auto [lhsElem0, lhsElem1, lhsElem2, lhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1, rhsElem2, rhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // TODO: Implement the real Karatsuba algorithm for 4x4 multiplication.
    // First part of Karatsuba algorithm
    auto z00 = b.create<mlir::arith::MulIOp>(lhsElem0, rhsElem0);
    auto z02 = b.create<mlir::arith::MulIOp>(lhsElem1, rhsElem1);
    auto z01_p1 = b.create<mlir::arith::AddIOp>(lhsElem0, lhsElem1);
    auto z01_p2 = b.create<mlir::arith::AddIOp>(rhsElem0, rhsElem1);
    auto z01_m = b.create<mlir::arith::MulIOp>(z01_p1, z01_p2);
    auto z01_s = b.create<mlir::arith::SubIOp>(z01_m, z00);
    auto z01 = b.create<mlir::arith::SubIOp>(z01_s, z02);

    // Second part I of Karatsuba algorithm
    auto z1a0 = b.create<mlir::arith::MulIOp>(lhsElem0, rhsElem2);
    auto z1a2 = b.create<mlir::arith::MulIOp>(lhsElem1, rhsElem3);
    auto z1a1_p1 = b.create<mlir::arith::AddIOp>(lhsElem0, lhsElem1);
    auto z1a1_p2 = b.create<mlir::arith::AddIOp>(rhsElem2, rhsElem3);
    auto z1a1_m = b.create<mlir::arith::MulIOp>(z1a1_p1, z1a1_p2);
    auto z1a1_s = b.create<mlir::arith::SubIOp>(z1a1_m, z1a0);
    auto z1a1 = b.create<mlir::arith::SubIOp>(z1a1_s, z1a2);

    // Second part II of Karatsuba algorithm
    auto z1b0 = b.create<mlir::arith::MulIOp>(lhsElem2, rhsElem0);
    auto z1b2 = b.create<mlir::arith::MulIOp>(lhsElem3, rhsElem1);
    auto z1b1_p1 = b.create<mlir::arith::AddIOp>(lhsElem2, lhsElem3);
    auto z1b1_p2 = b.create<mlir::arith::AddIOp>(rhsElem0, rhsElem1);
    auto z1b1_m = b.create<mlir::arith::MulIOp>(z1b1_p1, z1b1_p2);
    auto z1b1_s = b.create<mlir::arith::SubIOp>(z1b1_m, z1b0);
    auto z1b1 = b.create<mlir::arith::SubIOp>(z1b1_s, z1b2);

    auto out2Kara = b.create<mlir::arith::AddIOp>(z1a0, z1b0);
    auto out2Carry = b.create<mlir::arith::AddIOp>(out2Kara, z02);
    auto out3Carry = b.create<mlir::arith::AddIOp>(z1a1, z1b1);

    // Output are now all 16b elements, wants presentation of 4x8b
    auto output0Lsb = b.create<mlir::arith::TruncIOp>(realTy, z00);
    auto output0LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output0Lsb);
    auto output0Msb =
        b.create<mlir::arith::ShRUIOp>(z00, constantOp.getResult());

    auto output1Lsb = b.create<mlir::arith::TruncIOp>(realTy, z01);
    auto output1LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output1Lsb);
    auto output1Msb =
        b.create<mlir::arith::ShRUIOp>(z01, constantOp.getResult());

    auto output2Lsb = b.create<mlir::arith::TruncIOp>(realTy, out2Carry);
    auto output2LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output2Lsb);
    auto output2Msb =
        b.create<mlir::arith::ShRUIOp>(out2Carry, constantOp.getResult());

    auto output3Lsb = b.create<mlir::arith::TruncIOp>(realTy, out3Carry);
    auto output3LsbHigh = b.create<mlir::arith::ExtUIOp>(elemTy, output3Lsb);

    auto output1 = b.create<mlir::arith::AddIOp>(output1LsbHigh, output0Msb);
    auto output2 = b.create<mlir::arith::AddIOp>(output2LsbHigh, output1Msb);
    auto output3 = b.create<mlir::arith::AddIOp>(output3LsbHigh, output2Msb);

    Value resultVec = constructResultTensor(
        rewriter, loc, newTy, {output0LsbHigh, output1, output2, output3});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct ConvertArithConstant final
    : OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type oldType = op.getType();
    auto newType = getTypeConverter()->convertType<RankedTensorType>(oldType);

    if (!newType)
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("unsupported type: {0}", op.getType()));

    unsigned newBitWidth = newType.getElementTypeBitWidth();
    Attribute oldValue = op.getValueAttr();

    if (auto intAttr = dyn_cast<IntegerAttr>(oldValue)) {
      auto [low, midLow, midHigh, high] =
          getQuarters(intAttr.getValue(), newBitWidth);
      auto newAttr =
          DenseElementsAttr::get(newType, {low, midLow, midHigh, high});
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, newAttr);
      return success();
    }

    if (auto splatAttr = dyn_cast<SplatElementsAttr>(oldValue)) {
      auto [low, midLow, midHigh, high] =
          getQuarters(splatAttr.getSplatValue<APInt>(), newBitWidth);
      int64_t numSplatElems = splatAttr.getNumElements();
      SmallVector<APInt> values;
      values.reserve(numSplatElems * 4);
      for (int64_t i = 0; i < numSplatElems; ++i) {
        values.push_back(low);
        values.push_back(midLow);
        values.push_back(midHigh);
        values.push_back(high);
      }

      auto attr = DenseElementsAttr::get(newType, values);
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attr);
      return success();
    }

    if (auto elemsAttr = dyn_cast<DenseElementsAttr>(oldValue)) {
      int64_t numElems = elemsAttr.getNumElements();
      SmallVector<APInt> values;
      values.reserve(numElems * 4);
      for (const APInt &origVal : elemsAttr.getValues<APInt>()) {
        auto [low, midLow, midHigh, high] = getQuarters(origVal, newBitWidth);
        values.push_back(std::move(low));
        values.push_back(std::move(midLow));
        values.push_back(std::move(midHigh));
        values.push_back(std::move(high));
      }

      auto attr = DenseElementsAttr::get(newType, values);
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attr);
      return success();
    }

    return rewriter.notifyMatchFailure(op.getLoc(),
                                       "unhandled constant attribute");
  }
};

struct ConvertExtUI final : OpConversionPattern<mlir::arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ExtUIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());

    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Value resultVec = constructResultTensor(rewriter, loc, newTy, {op.getIn()});
    rewriter.replaceOp(op, resultVec);
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct QuarterWideInt : impl::QuarterWideIntBase<QuarterWideInt> {
  using QuarterWideIntBase::QuarterWideIntBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();
    RewritePatternSet patterns(context);
    QuarterWideTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };

    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      tensor::TensorDialect>(opLegalCallback);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    patterns.add<ConvertAddI, ConvertMulI, ConvertExtUI, ConvertArithConstant>(
        typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();

    // Remove the uncessary tensor ops between each converted arith operation.
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCSEPass());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace arith
}  // namespace heir
}  // namespace mlir
