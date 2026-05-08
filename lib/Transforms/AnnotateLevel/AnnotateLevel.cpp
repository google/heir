#include "lib/Transforms/AnnotateLevel/AnnotateLevel.h"

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "annotate-level"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATELEVEL
#include "lib/Transforms/AnnotateLevel/AnnotateLevel.h.inc"

struct AnnotateLevel : impl::AnnotateLevelBase<AnnotateLevel> {
  using AnnotateLevelBase::AnnotateLevelBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>(levelBudget);

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    walkValues(getOperation(), [&](Value value) {
      auto* lattice = solver.lookupState<LevelLattice>(value);
      if (!lattice) return;

      auto& state = lattice->getValue();
      if (!state.isInitialized()) return;

      OpBuilder b(value.getContext());
      Attribute attr;
      if (state.isInvalid()) {
        attr = b.getStringAttr("invalid");
      } else if (state.isMaxLevel()) {
        attr = b.getStringAttr("max");
      } else if (state.isInt()) {
        attr = b.getIndexAttr(state.getInt());
      }

      if (attr) {
        setAttributeAssociatedWith(value, "mgmt.level", attr);
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
