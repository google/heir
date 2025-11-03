#include <cstdint>
#include <utility>

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
#include "lib/Transforms/PopulateScale/PopulateScalePatterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

class CKKSAdjustScaleMaterializer : public AdjustScaleMaterializer {
 public:
  virtual ~CKKSAdjustScaleMaterializer() = default;

  int64_t deltaScale(int64_t scale, int64_t inputScale) const override {
    // High-precision scale management (#2364):
    // With actual scales (not log scales), delta is computed as scale /
    // inputScale For backward compatibility when scales are still in log
    // domain, fall back to subtraction This assumes scales are small enough to
    // fit in int64_t for now
    if (inputScale == 0) {
      return scale;
    }
    return scale / inputScale;
  }
};

#define GEN_PASS_DEF_POPULATESCALECKKS
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

struct PopulateScaleCKKS : impl::PopulateScaleCKKSBase<PopulateScaleCKKS> {
  using PopulateScaleCKKSBase::PopulateScaleCKKSBase;

  void runOnOperation() override {
    // skip scale management for openfhe
    if (moduleIsOpenfhe(getOperation())) {
      return;
    }

    auto ckksSchemeParamAttr = mlir::dyn_cast<ckks::SchemeParamAttr>(
        getOperation()->getAttr(ckks::CKKSDialect::kSchemeParamAttrName));
    auto logDefaultScale = ckksSchemeParamAttr.getLogDefaultScale();

    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    dataflow::loadBaselineAnalyses(solver);
    // ScaleAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // High-precision scale management (#2364): convert logDefaultScale to
    // actual scale inputScale = 2^logDefaultScale
    auto logScale = logDefaultScale;
    if (beforeMulIncludeFirstMul) {
      // encode at double degree
      logScale *= 2;
    }
    // Convert from log scale to actual scale using APInt
    // Need to compute 2^logScale
    // Use a bit width large enough to hold 2^logScale (logScale + 1 bits
    // minimum)
    unsigned bitWidth = std::max(64u, static_cast<unsigned>(logScale) + 1);
    auto inputScale = llvm::APInt(bitWidth, 1).shl(logScale);
    solver.load<ScaleAnalysis<CKKSScaleModel>>(
        ckks::SchemeParam::getSchemeParamFromAttr(ckksSchemeParamAttr),
        /*inputScale*/ inputScale);
    // Back-prop ScaleAnalysis depends on (forward) ScaleAnalysis
    solver.load<ScaleAnalysisBackward<CKKSScaleModel>>(
        symbolTable,
        ckks::SchemeParam::getSchemeParamFromAttr(ckksSchemeParamAttr));

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }
    // at this time all adjust_scale should have ScaleLattice for its result.
    // all plaintext (mgmt.init) should have ScaleLattice for its result.

    // pass scale to AnnotateMgmt pass
    annotateScale(getOperation(), &solver);
    OpPassManager annotateMgmt("builtin.module");
    annotateMgmt.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(annotateMgmt, getOperation());

    // convert adjust scale to mul plain
    RewritePatternSet patterns(&getContext());
    CKKSAdjustScaleMaterializer materializer;
    // TODO(#1641): handle arith.muli in CKKS
    patterns.add<ConvertAdjustScaleToMulPlain<arith::MulFOp>>(&getContext(),
                                                              &materializer);
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    // run canonicalizer and CSE to clean up arith.constant and move no-op out
    // of the secret.generic
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
