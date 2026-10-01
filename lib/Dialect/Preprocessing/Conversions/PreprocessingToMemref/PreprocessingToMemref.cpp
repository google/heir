#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToMemref/PreprocessingToMemref.h"

#include <utility>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DEF_PREPROCESSINGTOMEMREF
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToMemref/PreprocessingToMemref.h.inc"

namespace {

struct PreprocessingToMemref
    : impl::PreprocessingToMemrefBase<PreprocessingToMemref> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (!containsDialects<PreprocessingDialect>(module)) {
      return;
    }

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

    FlatMemrefPreprocessingTypeConverter typeConverter(analysis);

    ConversionTarget target(getContext());
    target.addIllegalDialect<PreprocessingDialect>();
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           affine::AffineDialect, func::FuncDialect>();

    RewritePatternSet patterns(&getContext());
    populatePreprocessingToFlatMemrefPatterns(typeConverter, patterns,
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
