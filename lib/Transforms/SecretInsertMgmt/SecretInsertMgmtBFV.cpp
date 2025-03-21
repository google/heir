#include <utility>

#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
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

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<MulDepthAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    RewritePatternSet patternsRelinearize(&getContext());
    patternsRelinearize.add<MultRelinearize<arith::MulIOp>>(
        &getContext(), getOperation(), &solver);
    // this line is not used by B/FV but used by CKKS.
    patternsRelinearize.add<MultRelinearize<arith::MulFOp>>(
        &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsRelinearize));

    auto maxMulDepth = getMaxMulDepth(getOperation(), solver);

    // annotate mgmt attribute with all levels set to mulDepth
    auto level = maxMulDepth;
    OpPassManager annotateMgmtPipeline("builtin.module");
    mgmt::AnnotateMgmtOptions annotateMgmtOptions;
    annotateMgmtOptions.baseLevel = level;
    annotateMgmtPipeline.addPass(mgmt::createAnnotateMgmt(annotateMgmtOptions));
    (void)runPipeline(annotateMgmtPipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
