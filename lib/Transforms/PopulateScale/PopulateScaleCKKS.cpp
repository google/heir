#include <cstdint>
#include <utility>

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Transforms/PopulateScale/PopulateScalePatterns.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
// IWYU pragma: end_keep

#define DEBUG_TYPE "populate-scale-ckks"

namespace mlir {
namespace heir {

class CKKSAdjustScaleMaterializer : public AdjustScaleMaterializer {
 public:
  virtual ~CKKSAdjustScaleMaterializer() = default;

  int64_t deltaScale(int64_t scale, int64_t inputScale) const override {
    // TODO(#1640): support high-precision scale management
    return scale - inputScale;
  }
};

#define GEN_PASS_DEF_POPULATESCALECKKS
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

LogicalResult createAndRunDataflow(Operation* op, DataFlowSolver& solver,
                                   int64_t logDefaultScale,
                                   ckks::SchemeParamAttr ckksSchemeParamAttr,
                                   bool beforeMulIncludeFirstMul) {
  SymbolTableCollection symbolTable;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  auto inputScale = logDefaultScale;
  if (beforeMulIncludeFirstMul) {
    LDBG() << "Encoding at scale^2 due to 'include-first-mul' config";
    inputScale *= 2;
  }
  auto param = ckks::getSchemeParamFromAttr(ckksSchemeParamAttr);
  solver.load<ScaleAnalysis<CKKSScaleModel>>(param,
                                             /*inputScale*/ inputScale);
  // Back-prop ScaleAnalysis depends on (forward) ScaleAnalysis
  solver.load<ScaleAnalysisBackward<CKKSScaleModel>>(symbolTable, param);

  return solver.initializeAndRun(op);
}

struct PopulateScaleCKKS : impl::PopulateScaleCKKSBase<PopulateScaleCKKS> {
  using PopulateScaleCKKSBase::PopulateScaleCKKSBase;

  void runOnOperation() override {
    auto ckksSchemeParamAttr = mlir::dyn_cast<ckks::SchemeParamAttr>(
        getOperation()->getAttr(ckks::CKKSDialect::kSchemeParamAttrName));
    auto logDefaultScale = ckksSchemeParamAttr.getLogDefaultScale();

    DataFlowSolver solver;
    if (failed(createAndRunDataflow(getOperation(), solver, logDefaultScale,
                                    ckksSchemeParamAttr,
                                    beforeMulIncludeFirstMul))) {
      signalPassFailure();
      return;
    }

    // at this time all adjust_scale should have ScaleLattice for its result.
    // all plaintext (mgmt.init) should have ScaleLattice for its result.
    // However, due to the naivete of the scale analysis, there can be
    // sections of IR in which no scale could be propagated through. E.g., an
    // op surrounded by two adjust_scale ops that block propagation. In this
    // case these adjust_scale ops should be removed.
    getOperation()->walk([&](mgmt::AdjustScaleOp op) {
      auto* lattice = solver.lookupState<ScaleLattice>(op.getResult());
      if (!lattice || !lattice->getValue().isInitialized()) {
        op.emitOpError() << "Dataflow analysis failed to populate scale "
                            "lattice for result\n";
        op->replaceAllUsesWith(ValueRange{op.getInput()});
        op->erase();
      }
    });

    DataFlowSolver solver2;
    if (failed(createAndRunDataflow(getOperation(), solver2, logDefaultScale,
                                    ckksSchemeParamAttr,
                                    beforeMulIncludeFirstMul))) {
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](mgmt::InitOp op) {
      auto* lattice = solver.lookupState<ScaleLattice>(op.getResult());
      if (!lattice || !lattice->getValue().isInitialized()) {
        op.emitOpError() << "Dataflow analysis failed to populate scale "
                            "lattice for result\n";
        signalPassFailure();
      }
    });

    LDBG() << "Running annotate-mgmt sub-pass";
    annotateScale(getOperation(), &solver2);
    OpPassManager annotateMgmt("builtin.module");
    annotateMgmt.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(annotateMgmt, getOperation());

    LLVM_DEBUG({
      llvm::dbgs() << "Dumping op after annotate-mgmt pass:\n";
      getOperation()->dump();
    });

    LDBG() << "convert adjust_scale to mul_plain";
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
