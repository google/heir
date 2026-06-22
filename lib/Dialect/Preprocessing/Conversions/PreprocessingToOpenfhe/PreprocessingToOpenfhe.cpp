#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToOpenfhe/PreprocessingToOpenfhe.h"

#include <utility>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Utils/ConversionUtils.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DEF_PREPROCESSINGTOOPENFHE
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToOpenfhe/PreprocessingToOpenfhe.h.inc"

namespace {

struct PreprocessingToOpenfhe
    : impl::PreprocessingToOpenfheBase<PreprocessingToOpenfhe> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    PreprocessingStorageLayoutAnalysis analysis(module);
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    if (analysis.getTotalSizes().empty()) {
      getOperation()->emitWarning()
          << "split-preprocessing was run, but preprocessing-to-openfhe "
             "determined there are no plaintexts to preprocess.";
      signalPassFailure();
      return;
    }

    Type targetType = openfhe::PlaintextType::get(&getContext());
    SingleMemrefPreprocessingTypeConverter typeConverter(analysis, targetType);

    ConversionTarget target(getContext());
    target.addIllegalDialect<PreprocessingDialect>();
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           affine::AffineDialect, func::FuncDialect,
                           openfhe::OpenfheDialect>();

    RewritePatternSet patterns(&getContext());
    populateCommonPreprocessingToMemrefPatterns(typeConverter, patterns);

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
