#include <cstdint>
#include <utility>

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/PlaintextParams.h"
#include "lib/Transforms/PopulateScale/PopulateScalePatterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
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

class PlaintextAdjustScaleMaterializer : public AdjustScaleMaterializer {
 public:
  virtual ~PlaintextAdjustScaleMaterializer() = default;

  int64_t deltaScale(int64_t scale, int64_t inputScale) const override {
    return scale - inputScale;
  }
};

#define GEN_PASS_DEF_POPULATESCALEPLAINTEXT
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

struct PopulateScalePlaintext
    : impl::PopulateScalePlaintextBase<PopulateScalePlaintext> {
  using PopulateScalePlaintextBase::PopulateScalePlaintextBase;

  void runOnOperation() override {
    auto logDefaultScaleAttr = mlir::dyn_cast_or_null<IntegerAttr>(
        getOperation()->getAttr(kPlaintextSchemeAttrName));
    if (!logDefaultScaleAttr) {
      getOperation()->emitOpError() << "Expected a " << kPlaintextSchemeAttrName
                                    << " attribute at the module level.\n";
      signalPassFailure();
      return;
    }
    int64_t logDefaultScale = logDefaultScaleAttr.getInt();
    PlaintextSchemeParam param(logDefaultScale);

    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    // ScaleAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    solver.load<ScaleAnalysis<PlaintextScaleModel>>(param, logDefaultScale);
    // Back-prop ScaleAnalysis depends on (forward) ScaleAnalysis
    solver.load<ScaleAnalysisBackward<PlaintextScaleModel>>(symbolTable, param);

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
    PlaintextAdjustScaleMaterializer materializer;
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
