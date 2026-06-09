#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToMemref/PreprocessingToMemref.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"      // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
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

#define GEN_PASS_DEF_PREPROCESSINGTOMEMREF
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToMemref/PreprocessingToMemref.h.inc"

namespace {

// Since a preprocessing.storage can store multiple element types, and this
// can be lowered to multiple distinct memrefs, we need to keep track of
// the mapping from the type to its corresponding memref. This is decided
// by the order of the types in the parameterized storage type.
FailureOr<int> getElementTypeIndex(
    preprocessing::PreprocessingStorageType storageTy, Type elementTy) {
  auto elementTypes = storageTy.getElementTypes();
  auto it = llvm::find(elementTypes, elementTy);
  if (it == elementTypes.end()) {
    // This should be caught by the validate-preprocessing pass.
    return failure();
  }
  return std::distance(elementTypes.begin(), it);
}

Value getValueOrCreateConstantIndexOp(OpBuilder& builder, Location loc,
                                      OpFoldResult ofr) {
  if (auto val = dyn_cast<Value>(ofr)) {
    return val;
  }
  int64_t intVal = cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
  return arith::ConstantIndexOp::create(builder, loc, intVal);
}

// Returns an list of enclosing loop ops (affine.for, LoopLikeOpInterface) in
// outermost-to-innermost order.
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

// PreprocessingToMemref converts a (globally unique) preprocessing.storage SSA
// value to one or more flat memrefs (one for each type).
//
// getLinearIndex takes as input the data about a particular preprocessing.store
// or preprocessing.load op, and outputs an SSA value corresponding to the index
// within the flat memref of that value.
//
// The PreprocessingStorageLayoutAnalysis calculates offsets for each site_id,
// which corresponds to a target encode op in the original IR. Given this base
// offset, the variadic indices are matched with induction variables of an
// enclosing loop nest, and flattened into an SSA value corresponding to a 1D
// absolute index. If there is no containing loop, nest the relative offset is
// 0. Otherwise it is a flattened (start, stop, step)-appropriate shift from
// the base offset for the given site_id.
FailureOr<Value> getLinearIndex(
    OpBuilder& builder, Location loc, Operation* op, uint32_t siteId,
    ValueRange indices, const PreprocessingStorageLayoutAnalysis& analysis) {
  SmallVector<Operation*, 4> enclosingLoops = getEnclosingLoopsOuterToInner(op);
  if (enclosingLoops.size() != indices.size()) {
    return op->emitOpError() << "Number of indices (" << indices.size()
                             << ") does not match number of enclosing loops ("
                             << enclosingLoops.size() << ")";
  }

  std::optional<SiteLayout> layout = analysis.getLayout(siteId);
  if (!layout.has_value()) {
    return op->emitOpError() << "Missing layout for site ID " << siteId;
  }
  int64_t baseOffset = layout->offset;

  Value linearIndex = arith::ConstantIndexOp::create(builder, loc, baseOffset);
  if (enclosingLoops.empty()) {
    // The op is not in a loop, so the base offset is the index of the memref
    return linearIndex;
  }

  int64_t currentStride = 1;
  SmallVector<Value, 4> lbs(indices.size());
  SmallVector<Value, 4> steps(indices.size());
  SmallVector<int64_t, 4> strides(indices.size());
  SmallVector<int64_t, 4> tripCounts(indices.size());

  // First calculate each loop's bounds, steps, strides, and trip counts.
  for (int i = indices.size() - 1; i >= 0; --i) {
    Operation* loop = enclosingLoops[i];
    Value lb;
    Value step;
    int64_t tripCount = 0;

    if (auto affineFor = dyn_cast<affine::AffineForOp>(loop)) {
      if (!affineFor.hasConstantLowerBound()) {
        return loop->emitOpError() << "Expected constant lower bound";
      }
      lb = arith::ConstantIndexOp::create(builder, loc,
                                          affineFor.getConstantLowerBound());
      int64_t stepVal = affineFor.getStepAsInt();
      step = arith::ConstantIndexOp::create(builder, loc, stepVal);
      std::optional<uint64_t> tc = affine::getConstantTripCount(affineFor);
      if (!tc.has_value()) {
        return loop->emitOpError() << "Expected constant trip count";
      }
      tripCount = *tc;
    } else if (auto loopLikeOp = dyn_cast<LoopLikeOpInterface>(loop)) {
      std::optional<OpFoldResult> ofrLb = loopLikeOp.getSingleLowerBound();
      std::optional<OpFoldResult> ofrStep = loopLikeOp.getSingleStep();
      if (!ofrLb || !ofrStep) {
        return loop->emitOpError() << "Expected single loop bounds and step";
      }
      lb = getValueOrCreateConstantIndexOp(builder, loc, *ofrLb);
      step = getValueOrCreateConstantIndexOp(builder, loc, *ofrStep);
      std::optional<APInt> tc = loopLikeOp.getStaticTripCount();
      if (!tc.has_value()) {
        return loop->emitOpError() << "Expected constant trip count";
      }
      tripCount = tc->getZExtValue();
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

  // Convert the statically inferred bounds and steps and strides into an SSA
  // value calculating the linear index.
  for (int i = indices.size() - 1; i >= 0; --i) {
    Value iv = indices[i];
    Value lb = lbs[i];
    Value step = steps[i];
    int64_t stride = strides[i];

    Value norm = iv;
    // If the lower bound is constant and zero, we can skip the sub op
    std::optional<int64_t> lbConst = getConstantIntValue(lb);
    if (!lbConst.has_value() || *lbConst != 0) {
      norm = arith::SubIOp::create(builder, loc, norm, lb);
    }

    // If the stride is constant and one, we can skip the floor div op
    std::optional<int64_t> stepConst = getConstantIntValue(step);
    if (!stepConst.has_value() || *stepConst != 1) {
      norm = arith::FloorDivSIOp::create(builder, loc, norm, step);
    }

    // If the stride is one, we can skip the mul op
    Value dimOffset = norm;
    if (stride > 1) {
      Value strideVal = arith::ConstantIndexOp::create(builder, loc, stride);
      dimOffset = arith::MulIOp::create(builder, loc, norm, strideVal);
    }

    linearIndex = arith::AddIOp::create(builder, loc, linearIndex, dimOffset);
  }

  return linearIndex;
}

// PreprocessingTypeConverter sets up a 1-to-N type conversion for
// PreprocessingStorageType. In the Preprocessing dialect, a single SSA value of
// PreprocessingStorageType conceptually holds multiple distinct storage buffers
// (one for each supported element type, i.e., plaintexts with different
// encodings). This converter maps that single PreprocessingStorageType to
// multiple distinct MemRefTypes (one flat memref per element type) sized
// according to the layout analysis. This allows subsequent rewrite patterns to
// operate on the specific underlying memref corresponding to the element type
// being stored or loaded.
class PreprocessingTypeConverter : public TypeConverter {
 public:
  explicit PreprocessingTypeConverter(
      const PreprocessingStorageLayoutAnalysis& analysis) {
    addConversion([](Type type) { return type; });
    addConversion([&analysis](preprocessing::PreprocessingStorageType type,
                              SmallVectorImpl<Type>& results) {
      for (Type elementTy : type.getElementTypes()) {
        int64_t totalSize = analysis.getTotalSize(elementTy).value_or(0);
        results.push_back(MemRefType::get({totalSize}, elementTy));
      }
      return success();
    });
  }
};

struct EmptyOpPattern : public OpConversionPattern<EmptyOp> {
  using OneToNOpAdaptor =
      typename OpConversionPattern<EmptyOp>::OneToNOpAdaptor;
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EmptyOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertType(op.getStorage().getType(),
                                               resultTypes))) {
      return failure();
    }
    SmallVector<Value> allocOps;
    for (Type memrefTy : resultTypes) {
      allocOps.push_back(memref::AllocOp::create(rewriter, op.getLoc(),
                                                 cast<MemRefType>(memrefTy)));
    }
    rewriter.replaceOpWithMultiple(op, {allocOps});
    return success();
  }
};

struct StoreOpPattern : public OpConversionPattern<StoreOp> {
  using OneToNOpAdaptor =
      typename OpConversionPattern<StoreOp>::OneToNOpAdaptor;
  StoreOpPattern(const TypeConverter& converter, MLIRContext* context,
                 const PreprocessingStorageLayoutAnalysis& analysis)
      : OpConversionPattern(converter, context), analysis(analysis) {}

  LogicalResult matchAndRewrite(
      StoreOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto storageTy = cast<PreprocessingStorageType>(op.getStorage().getType());
    FailureOr<int> elemIndex =
        getElementTypeIndex(storageTy, op.getElementType());
    if (failed(elemIndex)) return failure();

    SmallVector<Value> flatIndices;
    for (ValueRange r : adaptor.getIndices()) {
      if (r.size() != 1) {
        return op->emitOpError() << "Expected exactly one SSA value per index";
      }
      flatIndices.push_back(r.front());
    }

    FailureOr<Value> index = getLinearIndex(
        rewriter, op.getLoc(), op, op.getSiteId(), flatIndices, analysis);
    if (failed(index)) return failure();

    ValueRange storageValues = adaptor.getStorage();
    if (*elemIndex >= storageValues.size()) {
      return op->emitOpError() << "Storage index out of bounds";
    }
    Value targetMemref = storageValues[*elemIndex];
    memref::StoreOp storeOp = memref::StoreOp::create(
        rewriter, op.getLoc(), adaptor.getValue().front(), targetMemref,
        index.value());
    rewriter.replaceOp(op, storeOp);
    return success();
  }

 private:
  const PreprocessingStorageLayoutAnalysis& analysis;
};

struct LoadOpPattern : public OpConversionPattern<LoadOp> {
  using OneToNOpAdaptor = typename OpConversionPattern<LoadOp>::OneToNOpAdaptor;
  LoadOpPattern(const TypeConverter& converter, MLIRContext* context,
                const PreprocessingStorageLayoutAnalysis& analysis)
      : OpConversionPattern(converter, context), analysis(analysis) {}

  LogicalResult matchAndRewrite(
      LoadOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto storageTy = cast<PreprocessingStorageType>(op.getStorage().getType());
    FailureOr<int> elemIndex =
        getElementTypeIndex(storageTy, op.getElementType());
    if (failed(elemIndex)) return failure();

    SmallVector<Value> flatIndices;
    for (ValueRange r : adaptor.getIndices()) {
      if (r.size() != 1) {
        return op->emitOpError() << "Expected exactly one SSA value per index";
      }
      flatIndices.push_back(r.front());
    }

    FailureOr<Value> index = getLinearIndex(
        rewriter, op.getLoc(), op, op.getSiteId(), flatIndices, analysis);
    if (failed(index)) return failure();

    ValueRange storageValues = adaptor.getStorage();
    if (*elemIndex >= storageValues.size()) {
      return op->emitOpError() << "Storage index out of bounds";
    }
    Value targetMemref = storageValues[*elemIndex];
    memref::LoadOp loadOp = memref::LoadOp::create(rewriter, op.getLoc(),
                                                   targetMemref, index.value());
    rewriter.replaceOp(op, loadOp);
    return success();
  }

 private:
  const PreprocessingStorageLayoutAnalysis& analysis;
};

struct PreprocessingToMemref
    : impl::PreprocessingToMemrefBase<PreprocessingToMemref> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    PreprocessingStorageLayoutAnalysis analysis(module);
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    if (analysis.getTotalSizes().empty()) {
      getOperation()->emitOpError()
          << "split-preprocessing was run, but preprocessing-to-memref "
             "determined there are no plaintexts to preprocess.";
      signalPassFailure();
      return;
    }

    PreprocessingTypeConverter typeConverter(analysis);

    ConversionTarget target(getContext());
    target.addIllegalDialect<PreprocessingDialect>();
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           affine::AffineDialect, func::FuncDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<EmptyOpPattern>(typeConverter, &getContext());
    patterns.add<StoreOpPattern, LoadOpPattern>(typeConverter, &getContext(),
                                                analysis);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
