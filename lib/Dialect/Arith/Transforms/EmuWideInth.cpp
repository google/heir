#include "lib/Dialect/Arith/Transforms/EmuWideInth.h"

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

#define GEN_PASS_DEF_EMUWIDEINTH
#include "lib/Dialect/Arith/Transforms/Passes.h.inc"

static constexpr unsigned maxIntWidth = 16;

class EmuWideTypeConverter : public TypeConverter {
 public:
  EmuWideTypeConverter(MLIRContext *ctx) {
    // Allow unknown types.
    addConversion([](Type ty) -> std::optional<Type> { return ty; });

    // Scalar case.
    addConversion([](IntegerType ty) -> std::optional<Type> {
      unsigned width = ty.getWidth();
      if (width <= maxIntWidth) return ty;

      // i2N --> vector<2xiN>
      if (width == 2 * maxIntWidth)
        return RankedTensorType::get(
            4, IntegerType::get(ty.getContext(), maxIntWidth));

      return nullptr;
    });

    // Vector case.
    addConversion([](ShapedType ty) -> std::optional<Type> {
      auto intTy = dyn_cast<IntegerType>(ty.getElementType());
      if (!intTy) return ty;

      unsigned width = intTy.getWidth();
      if (width <= maxIntWidth) return ty;

      // vector<...xi2N> --> vector<...x2xiN>
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
      //   (i2N, i2N) -> i2N --> (vector<2xiN>, vector<2xiN>) -> vector<2xiN>
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

/// Returns N bottom and N top bits from `value`, where N = `newBitWidth`.
/// Treats `value` as a 2*N bits-wide integer.
/// The bottom bits are returned in the first pair element, while the top bits
/// in the second one.
static std::pair<APInt, APInt> getHalves(const APInt &value,
                                         unsigned newBitWidth) {
  APInt low = value.extractBits(newBitWidth, 0);
  APInt high = value.extractBits(newBitWidth, newBitWidth);
  return {std::move(low), std::move(high)};
}

/// Returns the type with the last (innermost) dimension reduced to x1.
/// Scalarizes 1D vector inputs to match how we extract/insert vector values,
/// e.g.:
///   - vector<3x2xi16> --> vector<3x1xi16>
///   - vector<2xi16>   --> i16
static Type reduceInnermostDim(RankedTensorType type) {
  if (type.getShape().size() == 1) return type.getElementType();

  auto newShape = to_vector(type.getShape());
  newShape.back() = 1;
  return RankedTensorType::get(newShape, type.getElementType());
}

/// Extracts the `input` vector slice with elements at the last dimension offset
/// by `lastOffset`. Returns a value of vector type with the last dimension
/// reduced to x1 or fully scalarized, e.g.:
///   - vector<3x2xi16> --> vector<3x1xi16>
///   - vector<2xi16>   --> i16
static Value extractLastDimSlice(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input,
                                 int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<RankedTensorType>(input.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Create valueRange
  auto intAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), lastOffset);
  auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
  ValueRange valueRange(constantOp.getResult());

  // Scalarize the result in case of 1D vectors.
  if (shape.size() == 1)
    return rewriter.create<tensor::ExtractOp>(loc, input, valueRange);

  SmallVector<int64_t> offsets(shape.size(), 0);
  offsets.back() = lastOffset;
  auto sizes = llvm::to_vector(shape);
  sizes.back() = 1;
  SmallVector<int64_t> strides(shape.size(), 1);

  return rewriter.create<vector::ExtractStridedSliceOp>(loc, input, offsets,
                                                        sizes, strides);
}

/// Extracts two vector slices from the `input` whose type is `vector<...x2T>`,
/// with the first element at offset 0 and the second element at offset 1.
static std::tuple<Value, Value, Value, Value> extractLastDimHalves(
    ConversionPatternRewriter &rewriter, Location loc, Value input) {
  return {extractLastDimSlice(rewriter, loc, input, 0),
          extractLastDimSlice(rewriter, loc, input, 1),
          extractLastDimSlice(rewriter, loc, input, 2),
          extractLastDimSlice(rewriter, loc, input, 3)};
}

// Performs a vector shape cast to drop the trailing x1 dimension. If the
// `input` is a scalar, this is a noop.
// static Value dropTrailingX1Dim(ConversionPatternRewriter &rewriter,
//                                Location loc, Value input) {
//   auto vecTy = dyn_cast<RankedTensorType>(input.getType());
//   if (!vecTy) return input;

//   // Shape cast to drop the last x1 dimension.
//   ArrayRef<int64_t> shape = vecTy.getShape();
//   assert(shape.size() >= 2 && "Expected vector with at list two dims");
//   assert(shape.back() == 1 && "Expected the last vector dim to be x1");

//   auto newVecTy =
//       RankedTensorType::get(shape.drop_back(), vecTy.getElementType());
//   return rewriter.create<vector::ShapeCastOp>(loc, newVecTy, input);
// }

/// Performs a vector shape cast to append an x1 dimension. If the
/// `input` is a scalar, this is a noop.
// static Value appendX1Dim(ConversionPatternRewriter &rewriter, Location loc,
//                          Value input) {
//   auto vecTy = dyn_cast<RankedTensorType>(input.getType());
//   if (!vecTy) return input;

//   // Add a trailing x1 dim.
//   auto newShape = llvm::to_vector(vecTy.getShape());
//   newShape.push_back(1);
//   auto newTy = RankedTensorType::get(newShape, vecTy.getElementType());
//   return rewriter.create<vector::ShapeCastOp>(loc, newTy, input);
// }

/// Inserts the `source` vector slice into the `dest` vector at offset
/// `lastOffset` in the last dimension. `source` can be a scalar when `dest` is
/// a 1D vector.
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

  // SmallVector<int64_t> offsets(shape.size(), 0);
  // offsets.back() = lastOffset;
  // SmallVector<int64_t> strides(shape.size(), 1);
  // SmallVector<int64_t> sizes(shape.size(), 1);

  // return rewriter.create<tensor::InsertSliceOp>(loc, source, dest,
  //                                                      offsets, sizes,
  //                                                      strides);
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

/// Constructs a new vector of type `resultType` by creating a series of
/// insertions of `resultComponents`, each at the next offset of the last vector
/// dimension.
/// When all `resultComponents` are scalars, the result type is `vector<NxT>`;
/// when `resultComponents` are `vector<...x1xT>`s, the result type is
/// `vector<...xNxT>`, where `N` is the number of `resultComponents`.
static Value constructResultVector(ConversionPatternRewriter &rewriter,
                                   Location loc, RankedTensorType resultType,
                                   ValueRange resultComponents) {
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  // (void)resultShape;
  // assert(!resultShape.empty() && "Result expected to have dimensions");
  // assert(resultShape.back() == static_cast<int64_t>(resultComponents.size())
  // &&
  //        "Wrong number of result components");

  llvm::dbgs() << "### resultShape " << resultShape.size() << "\n";
  Value resultVec = createScalarOrSplatConstant(rewriter, loc, resultType, 0);
  llvm::dbgs() << "### resultVec " << resultVec << "\n";
  for (auto [i, component] : llvm::enumerate(resultComponents))
    resultVec = insertLastDimSlice(rewriter, loc, component, resultVec, i);

  return resultVec;
}

struct ConvertAddI final : OpConversionPattern<mlir::arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::AddIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::dbgs() << "Lets start " << op.getType() << "\n";
    Location loc = op->getLoc();

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    llvm::dbgs() << "newTy " << newTy << "\n";
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newElemTy = reduceInnermostDim(newTy);
    llvm::dbgs() << "newElemTy " << newElemTy << "\n";

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

    Value resultVec = constructResultVector(rewriter, loc, newTy,
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
    llvm::dbgs() << "Lets start " << op.getType() << "\n";
    Location loc = op->getLoc();

    auto newTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    llvm::dbgs() << "newTy " << newTy << "\n";
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newElemTy = reduceInnermostDim(newTy);
    llvm::dbgs() << "newElemTy " << newElemTy << "\n";

    auto [lhsElem0, lhsElem1, lhsElem2, lhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1, rhsElem2, rhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // Create Constant
    auto intAttr = rewriter.getIntegerAttr(newElemTy, maxIntWidth >> 1);
    auto constantOp = rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);

    Value resultVec = constructResultVector(rewriter, loc, newTy,
                                            {lowSum0, high1, high2, high3});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct EmuWideInth : impl::EmuWideInthBase<EmuWideInth> {
  using EmuWideInthBase::EmuWideInthBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();
    RewritePatternSet patterns(context);
    EmuWideTypeConverter typeConverter(context);

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

    patterns.add<ConvertAddI>(typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();

    //   // FIXME: implement pass
    // patterns.add<ConvertAddI>(typeConverter, context);

    // (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace arith
}  // namespace heir
}  // namespace mlir
