#include "lib/Dialect/Preprocessing/Conversions/Util.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "llvm/include/llvm/ADT/DenseSet.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"      // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"    // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

SingleMemrefPreprocessingTypeConverter::SingleMemrefPreprocessingTypeConverter(
    const PreprocessingStorageLayoutAnalysis& analysis, Type targetType)
    : analysis(analysis), targetType(targetType) {
  addConversion([](Type type) { return type; });
  addConversion([this](preprocessing::PreprocessingStorageType type,
                       SmallVectorImpl<Type>& results) {
    DenseSet<Type> uniqueTypes(type.getElementTypes().begin(),
                               type.getElementTypes().end());
    if (uniqueTypes.size() != 1) {
      emitWarning(UnknownLoc::get(this->targetType.getContext()),
                  "SingleMemrefPreprocessingTypeConverter expects exactly one "
                  "unique element type in PreprocessingStorageType");
      return failure();
    }
    Type elementTy = *uniqueTypes.begin();
    std::optional<int64_t> totalSize = this->analysis.getTotalSize(elementTy);
    if (!totalSize.has_value()) return failure();
    results.push_back(MemRefType::get({*totalSize}, this->targetType));
    return success();
  });
}

int64_t SingleMemrefPreprocessingTypeConverter::getFlatBaseOffset(
    PreprocessingStorageType storageTy, Type elementType,
    uint32_t siteId) const {
  DenseSet<Type> uniqueTypes(storageTy.getElementTypes().begin(),
                             storageTy.getElementTypes().end());
  assert(uniqueTypes.size() == 1 && "expected single unique element type");
  assert(*uniqueTypes.begin() == elementType && "element type mismatch");
  auto layout = analysis.getLayout(elementType, siteId);
  return failed(layout) ? 0 : layout->offset;
}

namespace {

SmallVector<Operation*, 4> getEnclosingLoopsOuterToInner(Operation* op) {
  SmallVector<Operation*, 4> enclosingLoops;
  Operation* parent = op->getParentOp();
  while (parent) {
    if (isa<affine::AffineForOp, LoopLikeOpInterface>(parent)) {
      enclosingLoops.push_back(parent);
    }
    parent = parent->getParentOp();
  }
  std::reverse(enclosingLoops.begin(), enclosingLoops.end());
  return enclosingLoops;
}

struct LoopBounds {
  Value lb;
  Value ub;
  Value step;
};

FailureOr<LoopBounds> getSingleLoopBounds(mlir::LoopLikeOpInterface loopOp,
                                          OpBuilder builder, Location loc) {
  std::optional<SmallVector<OpFoldResult>> loBnds = loopOp.getLoopLowerBounds();
  std::optional<SmallVector<OpFoldResult>> upBnds = loopOp.getLoopUpperBounds();
  std::optional<SmallVector<OpFoldResult>> steps = loopOp.getLoopSteps();
  if (!loBnds || !upBnds || !steps) return failure();
  if (loBnds->size() != 1 || upBnds->size() != 1 || steps->size() != 1)
    return failure();
  LoopBounds bounds;
  bounds.lb = getValueOrCreateConstantIndexOp(builder, loc, loBnds->front());
  bounds.ub = getValueOrCreateConstantIndexOp(builder, loc, upBnds->front());
  bounds.step = getValueOrCreateConstantIndexOp(builder, loc, steps->front());
  return bounds;
}

struct CommonEmptyOpPattern : public OpConversionPattern<EmptyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EmptyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertType(op.getStorage().getType(),
                                               resultTypes))) {
      return failure();
    }
    if (resultTypes.size() != 1) {
      return op->emitOpError() << "Expected conversion to a single memref type";
    }
    Value allocOp = memref::AllocOp::create(rewriter, op.getLoc(),
                                            cast<MemRefType>(resultTypes[0]));
    rewriter.replaceOp(op, allocOp);
    return success();
  }
};

struct CommonStoreOpPattern : public OpConversionPattern<StoreOp> {
  CommonStoreOpPattern(const SingleMemrefPreprocessingTypeConverter& tc,
                       MLIRContext* context)
      : OpConversionPattern(tc, context), typeConverter(tc) {}

  LogicalResult matchAndRewrite(
      StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto storageTy = cast<PreprocessingStorageType>(op.getStorage().getType());

    SmallVector<Value> flatIndices(adaptor.getIndices());
    int64_t baseOffset = typeConverter.getFlatBaseOffset(
        storageTy, op.getElementType(), op.getSiteId());

    FailureOr<Value> index =
        getLinearIndex(rewriter, op.getLoc(), op, baseOffset, flatIndices);
    if (failed(index)) return failure();

    Value targetMemref = adaptor.getStorage();
    memref::StoreOp storeOp = memref::StoreOp::create(
        rewriter, op.getLoc(), adaptor.getValue(), targetMemref, index.value());
    rewriter.replaceOp(op, storeOp);
    return success();
  }

 private:
  const SingleMemrefPreprocessingTypeConverter& typeConverter;
};

struct CommonLoadOpPattern : public OpConversionPattern<LoadOp> {
  CommonLoadOpPattern(const SingleMemrefPreprocessingTypeConverter& tc,
                      MLIRContext* context)
      : OpConversionPattern(tc, context), typeConverter(tc) {}

  LogicalResult matchAndRewrite(
      LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto storageTy = cast<PreprocessingStorageType>(op.getStorage().getType());

    SmallVector<Value> flatIndices(adaptor.getIndices());
    int64_t baseOffset = typeConverter.getFlatBaseOffset(
        storageTy, op.getElementType(), op.getSiteId());

    FailureOr<Value> index =
        getLinearIndex(rewriter, op.getLoc(), op, baseOffset, flatIndices);
    if (failed(index)) return failure();

    Value targetMemref = adaptor.getStorage();
    memref::LoadOp loadOp = memref::LoadOp::create(rewriter, op.getLoc(),
                                                   targetMemref, index.value());
    rewriter.replaceOp(op, loadOp);
    return success();
  }

 private:
  const SingleMemrefPreprocessingTypeConverter& typeConverter;
};

struct ConvertStoreOp : public OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type newElementTy = getTypeConverter()->convertType(op.getElementType());
    if (!newElementTy) return failure();

    rewriter.replaceOpWithNewOp<StoreOp>(
        op, adaptor.getValue(), adaptor.getStorage(), adaptor.getIndices(),
        op.getSiteId(), newElementTy);
    return success();
  }
};

struct ConvertLoadOp : public OpConversionPattern<LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getType());
    if (!newResultTy) return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(op, newResultTy, adaptor.getStorage(),
                                        adaptor.getIndices(), op.getSiteId(),
                                        newResultTy);
    return success();
  }
};

struct ConvertEmptyOp : public OpConversionPattern<EmptyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EmptyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getType());
    if (!newResultTy) return failure();

    rewriter.replaceOpWithNewOp<EmptyOp>(op, newResultTy);
    return success();
  }
};

}  // namespace

FailureOr<Value> getLinearIndex(OpBuilder& builder, Location loc, Operation* op,
                                int64_t baseOffset, ValueRange indices) {
  SmallVector<Operation*, 4> enclosingLoops = getEnclosingLoopsOuterToInner(op);
  if (enclosingLoops.size() != indices.size()) {
    return op->emitOpError() << "Number of indices (" << indices.size()
                             << ") does not match number of enclosing loops ("
                             << enclosingLoops.size() << ")";
  }

  Value linearIndex = arith::ConstantIndexOp::create(builder, loc, baseOffset);
  if (enclosingLoops.empty()) {
    return linearIndex;
  }

  int64_t currentStride = 1;
  SmallVector<Value, 4> lbs(indices.size());
  SmallVector<Value, 4> steps(indices.size());
  SmallVector<int64_t, 4> strides(indices.size());
  SmallVector<int64_t, 4> tripCounts(indices.size());

  for (int i = indices.size() - 1; i >= 0; --i) {
    Operation* loop = enclosingLoops[i];
    Value lb;
    Value step;
    int64_t tripCount = 0;

    if (auto loopLikeOp = dyn_cast<LoopLikeOpInterface>(loop)) {
      FailureOr<LoopBounds> bounds =
          getSingleLoopBounds(loopLikeOp, builder, loc);
      if (failed(bounds))
        return loop->emitOpError("Expected single loop bounds and step");

      std::optional<APInt> tc = loopLikeOp.getStaticTripCount();
      if (!tc.has_value()) {
        return loop->emitOpError() << "Expected constant trip count";
      }
      int64_t tcI64 = tc->getZExtValue();

      lb = bounds->lb;
      step = bounds->step;
      tripCount = tcI64;
    } else {
      return loop->emitOpError() << "Unsupported loop op";
    }

    lbs[i] = lb;
    steps[i] = step;
    tripCounts[i] = tripCount;
    strides[i] = currentStride;

    if (__builtin_mul_overflow(currentStride, tripCount, &currentStride)) {
      return loop->emitOpError() << "Stride overflow";
    }
  }

  for (int i = indices.size() - 1; i >= 0; --i) {
    Value iv = indices[i];
    Value lb = lbs[i];
    Value step = steps[i];
    int64_t stride = strides[i];

    Value norm = iv;
    std::optional<int64_t> lbConst = getConstantIntValue(lb);
    if (!lbConst.has_value() || *lbConst != 0) {
      norm = arith::SubIOp::create(builder, loc, norm, lb);
    }

    std::optional<int64_t> stepConst = getConstantIntValue(step);
    if (!stepConst.has_value() || *stepConst != 1) {
      norm = arith::FloorDivSIOp::create(builder, loc, norm, step);
    }

    Value dimOffset = norm;
    if (stride > 1) {
      Value strideVal = arith::ConstantIndexOp::create(builder, loc, stride);
      dimOffset = arith::MulIOp::create(builder, loc, norm, strideVal);
    }

    linearIndex = arith::AddIOp::create(builder, loc, linearIndex, dimOffset);
  }

  return linearIndex;
}

void populateCommonPreprocessingToMemrefPatterns(
    const SingleMemrefPreprocessingTypeConverter& typeConverter,
    RewritePatternSet& patterns) {
  patterns.add<CommonEmptyOpPattern>(
      typeConverter, typeConverter.getTargetType().getContext());
  patterns.add<CommonStoreOpPattern, CommonLoadOpPattern>(
      typeConverter, typeConverter.getTargetType().getContext());
}

void populatePreprocessingConversions(RewritePatternSet& patterns,
                                      const TypeConverter& typeConverter,
                                      MLIRContext* context) {
  patterns.add<ConvertStoreOp, ConvertLoadOp, ConvertEmptyOp>(typeConverter,
                                                              context);
}

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
