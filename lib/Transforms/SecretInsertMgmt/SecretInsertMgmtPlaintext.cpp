#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTPLAINTEXT
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct SecretInsertMgmtPlaintext
    : impl::SecretInsertMgmtPlaintextBase<SecretInsertMgmtPlaintext> {
  using SecretInsertMgmtPlaintextBase::SecretInsertMgmtPlaintextBase;

  void runOnOperation() override {
    // hardcode some options from other passes; I'm not sure we really care
    // which values they take because this is not performance critical.
    bool afterMul = false;
    bool beforeMulIncludeFirstMul = false;
    bool includeFirstMul = false;

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<MulDepthAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // Handle plaintext operands. In the plaintext backend, these are
    // "cleartext" operands which must be encoded as plaintexts, so the name is
    // confusing.
    RewritePatternSet patternsPlaintext(&getContext());
    patternsPlaintext.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
                          UseInitOpForPlaintextOperand<arith::SubIOp>,
                          UseInitOpForPlaintextOperand<arith::MulIOp>,
                          UseInitOpForPlaintextOperand<arith::AddFOp>,
                          UseInitOpForPlaintextOperand<arith::SubFOp>,
                          UseInitOpForPlaintextOperand<arith::MulFOp>>(
        &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsPlaintext));

    RewritePatternSet patternsMultModReduce(&getContext());
    // tensor::ExtractOp = mulConst + rotate
    patternsMultModReduce
        .add<ModReduceBefore<arith::MulIOp>, ModReduceBefore<arith::MulFOp>,
             ModReduceBefore<tensor::ExtractOp>>(
            &getContext(), beforeMulIncludeFirstMul, getOperation(), &solver);
    patternsMultModReduce.add<ModReduceBefore<secret::YieldOp>>(
        &getContext(), includeFirstMul, getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(),
                               std::move(patternsMultModReduce));

    // when other binary op operands level mismatch
    //
    // See also MatchCrossLevel documentation
    int idCounter = 0;  // for making adjust_scale op different to avoid cse
    RewritePatternSet patternsAddModReduce(&getContext());
    patternsAddModReduce
        .add<MatchCrossLevel<arith::AddIOp>, MatchCrossLevel<arith::SubIOp>,
             MatchCrossLevel<arith::MulIOp>, MatchCrossLevel<arith::AddFOp>,
             MatchCrossLevel<arith::SubFOp>, MatchCrossLevel<arith::MulFOp>>(
            &getContext(), &idCounter, getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsAddModReduce));

    // when other binary op operands mulDepth mismatch
    // this only happen for before-mul but not include-first-mul case
    // at the first level, a Value can be both mulResult or not mulResult
    // we should match their scale by adding one adjust scale op
    //
    // See also MatchCrossMulDepth documentation
    if (!beforeMulIncludeFirstMul && !afterMul) {
      RewritePatternSet patternsMulDepth(&getContext());
      patternsMulDepth.add<
          MatchCrossMulDepth<arith::MulIOp>, MatchCrossMulDepth<arith::AddIOp>,
          MatchCrossMulDepth<arith::SubIOp>, MatchCrossMulDepth<arith::MulFOp>,
          MatchCrossMulDepth<arith::AddFOp>, MatchCrossMulDepth<arith::SubFOp>>(
          &getContext(), &idCounter, getOperation(), &solver);
      (void)walkAndApplyPatterns(getOperation(), std::move(patternsMulDepth));
    }

    // call Canonicalizer here because mgmt ops need to be ordered
    // call CSE here because there may be redundant mod reduce
    // one Value may get mod reduced multiple times in
    // multiple Uses
    //
    // also run annotate-mgmt for lowering
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    pipeline.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
