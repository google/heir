#include <cstdint>
#include <utility>

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
#include "lib/Transforms/PopulateScale/PopulateScalePatterns.h"
#include "lib/Utils/APIntUtils.h"
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

class BGVAdjustScaleMaterializer : public AdjustScaleMaterializer {
 public:
  BGVAdjustScaleMaterializer(int64_t plaintextModulus)
      : plaintextModulus(plaintextModulus) {}

  virtual ~BGVAdjustScaleMaterializer() = default;

  int64_t deltaScale(int64_t scale, int64_t inputScale) const override {
    auto inputScaleInverse =
        multiplicativeInverse(llvm::APInt(64, inputScale),
                              llvm::APInt(64, plaintextModulus))
            .getSExtValue();
    return (scale * inputScaleInverse) % plaintextModulus;
  }

 private:
  int64_t plaintextModulus = 0;
};

#define GEN_PASS_DEF_POPULATESCALEBGV
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

struct PopulateScaleBGV : impl::PopulateScaleBGVBase<PopulateScaleBGV> {
  using PopulateScaleBGVBase::PopulateScaleBGVBase;

  void runOnOperation() override {
    auto bgvSchemeParamAttr = mlir::dyn_cast<bgv::SchemeParamAttr>(
        getOperation()->getAttr(bgv::BGVDialect::kSchemeParamAttrName));
    auto t = bgvSchemeParamAttr.getPlaintextModulus();

    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    dataflow::loadBaselineAnalyses(solver);
    // ScaleAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // set input scale to 1, which is arbitrary.
    solver.load<ScaleAnalysis<BGVScaleModel>>(
        bgv::SchemeParam::getSchemeParamFromAttr(bgvSchemeParamAttr),
        /*inputScale*/ 1);
    // Back-prop ScaleAnalysis depends on (forward) ScaleAnalysis
    solver.load<ScaleAnalysisBackward<BGVScaleModel>>(
        symbolTable,
        bgv::SchemeParam::getSchemeParamFromAttr(bgvSchemeParamAttr));

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
    BGVAdjustScaleMaterializer materializer(t);
    patterns.add<ConvertAdjustScaleToMulPlain<arith::MulIOp>>(&getContext(),
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
