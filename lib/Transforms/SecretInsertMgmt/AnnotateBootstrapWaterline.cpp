#include "lib/Analysis/LevelAnalysis/BootstrapWaterlineAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"             // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "annotate-bootstrap-waterline"

namespace mlir {
class ModuleOp;
namespace heir {

using ::mlir::ModuleOp;

#define GEN_PASS_DEF_ANNOTATEBOOTSTRAPWATERLINE
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct AnnotateBootstrapWaterline
    : impl::AnnotateBootstrapWaterlineBase<AnnotateBootstrapWaterline> {
  using AnnotateBootstrapWaterlineBase::AnnotateBootstrapWaterlineBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<BootstrapWaterlineAnalysis>(bootstrapWaterline, levelBudget);

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    walkValues(getOperation(), [&](Value value) {
      auto* lattice = solver.lookupState<BootstrapWaterlineLattice>(value);
      if (!lattice) return;

      auto& state = lattice->getValue();
      if (!state.getLevelState().isInitialized()) return;

      OpBuilder b(value.getContext());
      Attribute levelAttr;
      auto levelState = state.getLevelState();
      if (levelState.isInvalid()) {
        levelAttr = b.getStringAttr("invalid");
      } else if (levelState.isMaxLevel()) {
        levelAttr = b.getStringAttr("max");
      } else if (levelState.isInt()) {
        levelAttr = b.getI64IntegerAttr(levelState.getInt());
      }

      if (levelAttr) {
        setAttributeAssociatedWith(value, "mgmt.bootstrap_waterline_level",
                                   levelAttr);
      }
      setAttributeAssociatedWith(value, "mgmt.needs_bootstrap",
                                 b.getBoolAttr(state.getNeedsBootstrap()));
    });
  }
};

}  // namespace heir
}  // namespace mlir
