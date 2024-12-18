#include "lib/Dialect/Arith/Transforms/QuarterWideInt.h"

#include "llvm/include/llvm/ADT/APInt.h"                // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

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

      return nullptr;
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

      return nullptr;
    });

    // Function case.
    addConversion([this](FunctionType ty) -> std::optional<Type> {
      // Convert inputs and results, e.g.:
      //   (i2N, i2N) -> i2N --> (tensor<4xiN>, tensor<4xiN>) -> tensor<4xiN>
      SmallVector<Type> inputs;
      if (failed(convertTypes(ty.getInputs(), inputs))) return nullptr;

      SmallVector<Type> results;
      if (failed(convertTypes(ty.getResults(), results))) return nullptr;

      return FunctionType::get(ty.getContext(), inputs, results);
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
static std::tuple<APInt, APInt, APInt, APInt> getQuarters(
    const APInt &value, unsigned newBitWidth) {
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
static Type reduceInnermostDim(RankedTensorType type) {
  if (type.getShape().size() == 1) return type.getElementType();

  auto newShape = to_vector(type.getShape());
  newShape.back() = 1;
  return RankedTensorType::get(newShape, type.getElementType());
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

  // Create valueRange
  auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
  auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
  ValueRange valueRange(constantOp.getResult());

  // Scalarize the result in case of 1D tensors.
  if (shape.size() == 1)
    return rewriter.create<tensor::ExtractOp>(loc, input, valueRange);
}

/// Extracts four tensor slices from the `input` whose type is `tensor<...x4T>`,
/// with the first element at offset 0, second element at offset 1 and so on.
static std::tuple<Value, Value, Value, Value> extractLastDimHalves(
    ConversionPatternRewriter &rewriter, Location loc, Value input) {
  return {extractLastDimSlice(rewriter, loc, input, 0),
          extractLastDimSlice(rewriter, loc, input, 1),
          extractLastDimSlice(rewriter, loc, input, 2),
          extractLastDimSlice(rewriter, loc, input, 3)};
}

/// Inserts the `source` tensor slice into the `dest` tensor at offset
/// `lastOffset` in the last dimension. `source` can be a scalar when `dest` is
/// a 1D tensor.
static Value insertLastDimSlice(ConversionPatternRewriter &rewriter,
                                Location loc, Value source, Value dest,
                                int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<RankedTensorType>(dest.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Handle scalar source.
  if (isa<IntegerType>(source.getType())) {
    auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
    auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
    ValueRange valueRange(constantOp.getResult());

    return rewriter.create<tensor::InsertOp>(loc, source, dest, valueRange);
  }

  rewriter.notifyMatchFailure(
      loc, llvm::formatv("Problem with the insertions of obj {0}",
                         source.getType()));
}

static Value createScalarOrSplatConstant(OpBuilder &builder, Location loc,
                                         Type type, int64_t value) {
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
static Value constructResultTensor(ConversionPatternRewriter &rewriter,
                                   Location loc, RankedTensorType resultType,
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

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newElemTy = reduceInnermostDim(newTy);

    auto [lhsElem0, lhsElem1, lhsElem2, lhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1, rhsElem2, rhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // Create Constant
    auto intAttr = rewriter.getIntegerAttr(newElemTy, maxIntWidth >> 1);
    auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);

    auto lowSum0 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem0, rhsElem0);
    auto lowSum1 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem1, rhsElem1);
    auto lowSum2 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem2, rhsElem2);
    auto lowSum3 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem3, rhsElem3);

    auto carry0 = rewriter.create<mlir::arith::ShRSIOp>(loc, lowSum0,
                                                        constantOp.getResult());
    auto carry1 = rewriter.create<mlir::arith::ShRSIOp>(loc, lowSum1,
                                                        constantOp.getResult());
    auto carry2 = rewriter.create<mlir::arith::ShRSIOp>(loc, lowSum2,
                                                        constantOp.getResult());

    auto high1 = rewriter.create<mlir::arith::AddIOp>(loc, lowSum1, carry0);
    auto high2 = rewriter.create<mlir::arith::AddIOp>(loc, lowSum2, carry1);
    auto high3 = rewriter.create<mlir::arith::AddIOp>(loc, lowSum3, carry2);

    Value resultVec = constructResultTensor(rewriter, loc, newTy,
                                            {lowSum0, high1, high2, high3});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct ConvertMulI final : OpConversionPattern<mlir::arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::MulIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    auto [lhsElem0, lhsElem1, lhsElem2, lhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1, rhsElem2, rhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // FIXME: Implement the real Karatsuba algorithm for 4x4 multiplication.
    // First part of Karatsuba algorithm
    auto z00 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem0, rhsElem0);
    auto z02 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem1, rhsElem1);
    auto z01_p1 = rewriter.create<mlir::arith::AddIOp>(loc, lhsElem0, lhsElem1);
    auto z01_p2 = rewriter.create<mlir::arith::AddIOp>(loc, rhsElem0, rhsElem1);
    auto z01_m = rewriter.create<mlir::arith::MulIOp>(loc, z01_p1, z01_p2);
    auto z01_s = rewriter.create<mlir::arith::SubIOp>(loc, z01_m, z00);
    auto z01 = rewriter.create<mlir::arith::SubIOp>(loc, z01_s, z02);

    // Second part I of Karatsuba algorithm
    auto z1a0 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem0, rhsElem2);
    auto z1a2 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem1, rhsElem3);
    auto z1a1_p1 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem0, lhsElem1);
    auto z1a1_p2 =
        rewriter.create<mlir::arith::AddIOp>(loc, rhsElem2, rhsElem3);
    auto z1a1_m = rewriter.create<mlir::arith::MulIOp>(loc, z1a1_p1, z1a1_p2);
    auto z1a1_s = rewriter.create<mlir::arith::SubIOp>(loc, z1a1_m, z1a0);
    auto z1a1 = rewriter.create<mlir::arith::SubIOp>(loc, z1a1_s, z1a2);

    // Second part II of Karatsuba algorithm
    auto z1b0 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem2, rhsElem0);
    auto z1b2 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem3, rhsElem1);
    auto z1b1_p1 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem2, lhsElem3);
    auto z1b1_p2 =
        rewriter.create<mlir::arith::AddIOp>(loc, rhsElem0, rhsElem1);
    auto z1b1_m = rewriter.create<mlir::arith::MulIOp>(loc, z1b1_p1, z1b1_p2);
    auto z1b1_s = rewriter.create<mlir::arith::SubIOp>(loc, z1b1_m, z1b0);
    auto z1b1 = rewriter.create<mlir::arith::SubIOp>(loc, z1b1_s, z1b2);

    auto output2pre = rewriter.create<mlir::arith::AddIOp>(loc, z1a0, z1b0);
    auto output2 = rewriter.create<mlir::arith::AddIOp>(loc, output2pre, z02);
    auto output3 = rewriter.create<mlir::arith::AddIOp>(loc, z1a1, z1b1);

    Value resultVec = constructResultTensor(rewriter, loc, newTy,
                                            {z00, z01, output2, output3});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct ConvertMulIExt final
    : OpConversionPattern<mlir::arith::MulUIExtendedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::MulUIExtendedOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto newTy = RankedTensorType::get(
        7, IntegerType::get(op.getContext(), maxIntWidth));

    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType(0)));

    auto [lhsElem0, lhsElem1, lhsElem2, lhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1, rhsElem2, rhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // FIXME: Implement the real Karatsuba algorithm for 4x4 multiplication.
    // First part of Karatsuba algorithm
    auto z00 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem0, rhsElem0);
    auto z02 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem1, rhsElem1);
    auto z01_p1 = rewriter.create<mlir::arith::AddIOp>(loc, lhsElem0, lhsElem1);
    auto z01_p2 = rewriter.create<mlir::arith::AddIOp>(loc, rhsElem0, rhsElem1);
    auto z01_m = rewriter.create<mlir::arith::MulIOp>(loc, z01_p1, z01_p2);
    auto z01_s = rewriter.create<mlir::arith::SubIOp>(loc, z01_m, z00);
    auto z01 = rewriter.create<mlir::arith::SubIOp>(loc, z01_s, z02);

    // Second part I of Karatsuba algorithm
    auto z1a0 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem0, rhsElem2);
    auto z1a2 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem1, rhsElem3);
    auto z1a1_p1 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem0, lhsElem1);
    auto z1a1_p2 =
        rewriter.create<mlir::arith::AddIOp>(loc, rhsElem2, rhsElem3);
    auto z1a1_m = rewriter.create<mlir::arith::MulIOp>(loc, z1a1_p1, z1a1_p2);
    auto z1a1_s = rewriter.create<mlir::arith::SubIOp>(loc, z1a1_m, z1a0);
    auto z1a1 = rewriter.create<mlir::arith::SubIOp>(loc, z1a1_s, z1a2);

    // Second part II of Karatsuba algorithm
    auto z1b0 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem2, rhsElem0);
    auto z1b2 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem3, rhsElem1);
    auto z1b1_p1 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem2, lhsElem3);
    auto z1b1_p2 =
        rewriter.create<mlir::arith::AddIOp>(loc, rhsElem0, rhsElem1);
    auto z1b1_m = rewriter.create<mlir::arith::MulIOp>(loc, z1b1_p1, z1b1_p2);
    auto z1b1_s = rewriter.create<mlir::arith::SubIOp>(loc, z1b1_m, z1b0);
    auto z1b1 = rewriter.create<mlir::arith::SubIOp>(loc, z1b1_s, z1b2);

    // Third part of Karatsuba algorithm
    auto z20 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem2, rhsElem2);
    auto z22 = rewriter.create<mlir::arith::MulIOp>(loc, lhsElem3, rhsElem3);
    auto z21_p1 = rewriter.create<mlir::arith::AddIOp>(loc, lhsElem2, lhsElem3);
    auto z21_p2 = rewriter.create<mlir::arith::AddIOp>(loc, rhsElem2, rhsElem3);
    auto z21_m = rewriter.create<mlir::arith::MulIOp>(loc, z21_p1, z21_p2);
    auto z21_s = rewriter.create<mlir::arith::SubIOp>(loc, z21_m, z20);
    auto z21 = rewriter.create<mlir::arith::SubIOp>(loc, z21_s, z22);

    auto output2pre = rewriter.create<mlir::arith::AddIOp>(loc, z1a0, z1b0);
    auto output2 = rewriter.create<mlir::arith::AddIOp>(loc, output2pre, z02);

    auto output3 = rewriter.create<mlir::arith::AddIOp>(loc, z1a1, z1b1);

    auto output4pre = rewriter.create<mlir::arith::AddIOp>(loc, z1a2, z1b2);
    auto output4 = rewriter.create<mlir::arith::AddIOp>(loc, output4pre, z20);

    Value resultVec = constructResultTensor(
        rewriter, loc, newTy, {z00, z01, output2, output3, output4, z21, z22});
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

struct ConvertExtSI final : OpConversionPattern<mlir::arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::ExtSIOp op, OpAdaptor adaptor,
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
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      tensor::TensorDialect>(opLegalCallback);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    patterns.add<ConvertAddI, ConvertMulI, ConvertExtSI, ConvertMulIExt,
                 ConvertArithConstant>(typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace arith
}  // namespace heir
}  // namespace mlir
