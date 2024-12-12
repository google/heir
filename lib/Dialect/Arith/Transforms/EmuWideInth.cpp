#include "lib/Dialect/Arith/Transforms/EmuWideInth.h"

#include "llvm/include/llvm/ADT/APInt.h"               // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
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

/// Extracts the `input` vector slice with elements at the last dimension offset
/// by `lastOffset`. Returns a value of vector type with the last dimension
/// reduced to x1 or fully scalarized, e.g.:
///   - vector<3x2xi16> --> vector<3x1xi16>
///   - vector<2xi16>   --> i16
static Value extractLastDimSlice(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input,
                                 int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<VectorType>(input.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Scalarize the result in case of 1D vectors.
  if (shape.size() == 1)
    return rewriter.create<vector::ExtractOp>(loc, input, lastOffset);

  SmallVector<int64_t> offsets(shape.size(), 0);
  offsets.back() = lastOffset;
  auto sizes = llvm::to_vector(shape);
  sizes.back() = 1;
  SmallVector<int64_t> strides(shape.size(), 1);

  return rewriter.create<vector::ExtractStridedSliceOp>(loc, input, offsets,
                                                        sizes, strides);
}

/// Extracts four vector slices from the `input` whose type is `vector<...x4T>`,
/// with the first element at offset 0 and the second element at offset 1 and so
/// on.
static std::tuple<Value, Value, Value, Value> extractLastDimHalves(
    ConversionPatternRewriter &rewriter, Location loc, Value input) {
  return {extractLastDimSlice(rewriter, loc, input, 0),
          extractLastDimSlice(rewriter, loc, input, 1),
          extractLastDimSlice(rewriter, loc, input, 2),
          extractLastDimSlice(rewriter, loc, input, 3)};
}

/// Returns the type with the last (innermost) dimension reduced to x1.
/// Scalarizes 1D vector inputs to match how we extract/insert vector values,
/// e.g.:
///   - vector<3x2xi16> --> vector<3x1xi16>
///   - vector<2xi16>   --> i16
static Type reduceInnermostDim(VectorType type) {
  if (type.getShape().size() == 1) return type.getElementType();

  auto newShape = to_vector(type.getShape());
  newShape.back() = 1;
  return VectorType::get(newShape, type.getElementType());
}

/// Inserts the `source` vector slice into the `dest` vector at offset
/// `lastOffset` in the last dimension. `source` can be a scalar when `dest` is
/// a 1D vector.
static Value insertLastDimSlice(ConversionPatternRewriter &rewriter,
                                Location loc, Value source, Value dest,
                                int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<VectorType>(dest.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Handle scalar source.
  if (isa<IntegerType>(source.getType()))
    return rewriter.create<vector::InsertOp>(loc, source, dest, lastOffset);

  SmallVector<int64_t> offsets(shape.size(), 0);
  offsets.back() = lastOffset;
  SmallVector<int64_t> strides(shape.size(), 1);
  return rewriter.create<vector::InsertStridedSliceOp>(loc, source, dest,
                                                       offsets, strides);
}

static Value createScalarOrSplatConstant(OpBuilder &builder, Location loc,
                                         Type type, int64_t value) {
  unsigned elementBitWidth;
  if (auto intTy = dyn_cast<IntegerType>(type))
    elementBitWidth = intTy.getWidth();
  else
    elementBitWidth = cast<ShapedType>(type).getElementTypeBitWidth();

  TypedAttr attr;
  if (isa<IntegerType>(type)) {
    attr = builder.getIntegerAttr(type, value);
  } else {
    auto vecTy = cast<ShapedType>(type);
    attr = SplatElementsAttr::get(vecTy, value);
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
                                   Location loc, VectorType resultType,
                                   ValueRange resultComponents) {
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  (void)resultShape;
  assert(!resultShape.empty() && "Result expected to have dimensions");
  assert(resultShape.back() == static_cast<int64_t>(resultComponents.size()) &&
         "Wrong number of result components");

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
    auto newTy = getTypeConverter()->convertType<VectorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newElemTy = reduceInnermostDim(newTy);

    auto [lhsElem0, lhsElem1, lhsElem2, lhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1, rhsElem2, rhsElem3] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
        loc, newElemTy, rewriter.getIntegerAttr(newElemTy, 8));

    auto partSum0 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem0, rhsElem0);
    auto partSum1 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem1, rhsElem1);
    auto partSum2 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem2, rhsElem2);
    auto partSum3 =
        rewriter.create<mlir::arith::AddIOp>(loc, lhsElem3, rhsElem3);

    auto overflowVal0 = rewriter.create<mlir::arith::ShRSIOp>(
        loc, newElemTy, partSum0, shiftAmount);
    auto overflowVal1 = rewriter.create<mlir::arith::ShRSIOp>(
        loc, newElemTy, partSum1, shiftAmount);
    auto overflowVal2 = rewriter.create<mlir::arith::ShRSIOp>(
        loc, newElemTy, partSum2, shiftAmount);

    auto fullSum1 =
        rewriter.create<mlir::arith::AddIOp>(loc, partSum1, overflowVal0);
    auto fullSum2 =
        rewriter.create<mlir::arith::AddIOp>(loc, partSum2, overflowVal1);
    auto fullSum3 =
        rewriter.create<mlir::arith::AddIOp>(loc, partSum3, overflowVal2);

    Value resultVec = constructResultVector(
        rewriter, loc, newTy, {partSum0, fullSum1, fullSum2, fullSum3});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

struct EmuWideInth : impl::EmuWideInthBase<EmuWideInth> {
  using EmuWideInthBase::EmuWideInthBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // FIXME: implement pass
    patterns.add<ConvertAddI>(context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace arith
}  // namespace heir
}  // namespace mlir
