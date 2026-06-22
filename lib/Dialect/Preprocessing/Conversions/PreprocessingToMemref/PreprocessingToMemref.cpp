#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToMemref/PreprocessingToMemref.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"    // from @llvm-project
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

    std::optional<SiteLayout> layout =
        analysis.getLayout(op.getElementType(), op.getSiteId());
    if (!layout.has_value()) {
      return op->emitOpError()
             << "Missing layout for site ID " << op.getSiteId();
    }
    int64_t baseOffset = layout->offset;

    FailureOr<Value> index = preprocessing::getLinearIndex(
        rewriter, op.getLoc(), op, baseOffset, flatIndices);
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

    std::optional<SiteLayout> layout =
        analysis.getLayout(op.getElementType(), op.getSiteId());
    if (!layout.has_value()) {
      return op->emitOpError()
             << "Missing layout for site ID " << op.getSiteId();
    }
    int64_t baseOffset = layout->offset;

    FailureOr<Value> index = preprocessing::getLinearIndex(
        rewriter, op.getLoc(), op, baseOffset, flatIndices);
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
