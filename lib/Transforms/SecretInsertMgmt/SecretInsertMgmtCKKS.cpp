#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Mgmt/Transforms/Passes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt-ckks"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTCKKS
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct SecretInsertMgmtCKKS
    : impl::SecretInsertMgmtCKKSBase<SecretInsertMgmtCKKS> {
  using SecretInsertMgmtCKKSBase::SecretInsertMgmtCKKSBase;

  void runOnOperation() override {
    // Helper for future lowerings that want to know what scheme was used
    moduleSetCKKS(getOperation());

    bool canEmitAdjustScale = true;
    if (auto target = getTargetConfig(getOperation()); succeeded(target))
      canEmitAdjustScale = target->supportsAdjustScale;

    InsertMgmtPipelineOptions options;
    options.includeFloats = true;
    options.levelBudget = levelBudget;
    // A target with one canonical scale per level must rescale immediately
    // after multiplication. This turns scale mismatches into cross-level
    // mismatches that can be resolved with a level reduction.
    options.modReduceAfterMul = afterMul || !canEmitAdjustScale;
    options.modReduceBeforeMulIncludeFirstMul = beforeMulIncludeFirstMul;
    options.bootstrapWaterline = bootstrapWaterline;
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
    // 3. Canonicalizer will remove mgmt.level_reduce_min ops since now the
    //    level information is concrete.
    // 4. AnnotateMgmt will merge level and dimension into MgmtAttr, for further
    //   lowering.
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    mgmt::AnnotateMgmtOptions annotateOptions;
    annotateOptions.levelBudget = levelBudget;
    pipeline.addPass(mgmt::createAnnotateMgmt(annotateOptions));
    (void)runPipeline(pipeline, getOperation());

    if (!canEmitAdjustScale) {
      bool foundUnsupportedOp = false;
      getOperation()->walk([&](mgmt::AdjustScaleOp op) {
        op.emitError("target does not support mgmt.adjust_scale");
        foundUnsupportedOp = true;
      });
      if (foundUnsupportedOp) signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
