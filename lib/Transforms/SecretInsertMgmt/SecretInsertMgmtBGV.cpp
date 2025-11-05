#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Mgmt/Transforms/Passes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt-bgv"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTBGV
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct SecretInsertMgmtBGV
    : impl::SecretInsertMgmtBGVBase<SecretInsertMgmtBGV> {
  using SecretInsertMgmtBGVBase::SecretInsertMgmtBGVBase;

  void runOnOperation() override {
    // for Openfhe, use B/FV style mgmt: only relinearize, no level management.
    // still maintain the maximal level information though for lowering.
    if (moduleIsOpenfhe(getOperation())) {
      OpPassManager pipeline("builtin.module");
      pipeline.addPass(createSecretInsertMgmtBFV());
      (void)runPipeline(pipeline, getOperation());
      moduleSetBGV(getOperation());
      return;
    }

    // Helper for future lowerings that want to know what scheme was used
    moduleSetBGV(getOperation());

    if (afterMul && beforeMulIncludeFirstMul) {
      getOperation()->emitOpError()
          << "afterMul and beforeMulIncludeFirstMul cannot be true at the same "
             "time.\n";
      signalPassFailure();
      return;
    }

    InsertMgmtPipelineOptions options;
    options.includeFloats = true;
    options.modReduceAfterMul = afterMul;
    options.modReduceBeforeMulIncludeFirstMul = beforeMulIncludeFirstMul;
    LogicalResult result = runInsertMgmtPipeline(getOperation(), options);

    if (failed(result)) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Post secret-insert-mgmt pipeline cleanup\n");

    // 1. Canonicalizer reorders mgmt ops like Rescale/LevelReduce/AdjustScale.
    //    This is important for AnnotateMgmt.
    //    Canonicalizer also moves mgmt::InitOp out of secret.generic.
    // 2. CSE removes redundant mgmt::ModReduceOp.
    // 3. AnnotateMgmt will merge level and dimension into MgmtAttr, for further
    //   lowering.
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    pipeline.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
