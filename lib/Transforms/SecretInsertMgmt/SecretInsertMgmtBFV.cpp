#include <utility>

#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "mlir/include/mlir/Support/LLVM.h"       // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTBFV
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct SecretInsertMgmtBFV
    : impl::SecretInsertMgmtBFVBase<SecretInsertMgmtBFV> {
  using SecretInsertMgmtBFVBase::SecretInsertMgmtBFVBase;

  void runOnOperation() override {
    OpPassManager pipeline("builtin.module");
    SecretInsertMgmtBGVOptions options;
    options.includeFirstMul = false;
    pipeline.addPass(createSecretInsertMgmtBGV(options));
    (void)runPipeline(pipeline, getOperation());

    // Helper for future lowerings that want to know what scheme was used.
    // Should be called after the secret-insert-mgmt-bgv pass has been run.
    moduleSetBFV(getOperation());

    // inherit mulDepth information from BGV pass.
    mgmt::MgmtAttr mgmtAttr = nullptr;
    getOperation()->walk([&](secret::GenericOp op) {
      for (auto i = 0; i != op->getBlock()->getNumArguments(); ++i) {
        if ((mgmtAttr = dyn_cast<mgmt::MgmtAttr>(
                 op.getOperandAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName)))) {
          break;
        }
      }
    });

    if (!mgmtAttr) {
      getOperation()->emitError("No mgmt attribute found in the module");
      return signalPassFailure();
    }

    // Remove mgmt::ModReduceOp
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveOp<mgmt::ModReduceOp>>(&getContext());
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));

    // annotate mgmt attribute with all levels set to mulDepth
    auto level = mgmtAttr.getLevel();
    OpPassManager annotateMgmtPipeline("builtin.module");
    mgmt::AnnotateMgmtOptions annotateMgmtOptions;
    annotateMgmtOptions.baseLevel = level;
    annotateMgmtPipeline.addPass(mgmt::createAnnotateMgmt(annotateMgmtOptions));
    (void)runPipeline(annotateMgmtPipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
