#include <utility>

#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTBFV
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct SecretInsertMgmtBFV
    : impl::SecretInsertMgmtBFVBase<SecretInsertMgmtBFV> {
  using SecretInsertMgmtBFVBase::SecretInsertMgmtBFVBase;

  void runOnOperation() override {
    // Helper for future lowerings that want to know what scheme was used.
    moduleSetBFV(getOperation());

    int64_t maxMulDepth;
    // if the module is annotated with SchemeParamAttr, do not run analyses
    bool hasSchemeParam = false;
    if (auto schemeParam = getOperation()->getAttrOfType<bgv::SchemeParamAttr>(
            bgv::BGVDialect::kSchemeParamAttrName)) {
      hasSchemeParam = true;
      maxMulDepth = schemeParam.getQ().size() - 1;
    } else if (auto schemeParam =
                   getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
                       ckks::CKKSDialect::kSchemeParamAttrName)) {
      hasSchemeParam = true;
      maxMulDepth = schemeParam.getQ().size() - 1;
    }

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    if (!hasSchemeParam) {
      // try our best to analyse mul depth
      solver.load<MulDepthAnalysis>();
    }

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    if (!hasSchemeParam) {
      maxMulDepth = getMaxMulDepth(getOperation(), solver);
    }

    // handle plaintext operands
    RewritePatternSet patternsPlaintext(&getContext());
    patternsPlaintext.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
                          UseInitOpForPlaintextOperand<arith::SubIOp>,
                          UseInitOpForPlaintextOperand<arith::MulIOp>,
                          // these lines are not used by B/FV but used by CKKS.
                          UseInitOpForPlaintextOperand<arith::AddFOp>,
                          UseInitOpForPlaintextOperand<arith::SubFOp>,
                          UseInitOpForPlaintextOperand<arith::MulFOp>>(
        &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsPlaintext));

    RewritePatternSet patternsRelinearize(&getContext());
    patternsRelinearize.add<MultRelinearize<arith::MulIOp>,
                            // this line is not used by B/FV but used by CKKS.
                            MultRelinearize<arith::MulFOp>>(
        &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsRelinearize));

    auto level = maxMulDepth;
    // call Canonicalizer here because mgmt.init ops need to be moved out of the
    // secret.generic.
    // annotate mgmt attribute with all levels set to mulDepth
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    mgmt::AnnotateMgmtOptions annotateMgmtOptions;
    annotateMgmtOptions.baseLevel = level;
    pipeline.addPass(mgmt::createAnnotateMgmt(annotateMgmtOptions));
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
