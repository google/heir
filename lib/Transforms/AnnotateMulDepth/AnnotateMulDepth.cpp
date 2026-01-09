#include "lib/Transforms/AnnotateMulDepth/AnnotateMulDepth.h"

#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
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

#define DEBUG_TYPE "annotate-muldepth"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATEMULDEPTH
#include "lib/Transforms/AnnotateMulDepth/AnnotateMulDepth.h.inc"

struct AnnotateMulDepth : impl::AnnotateMulDepthBase<AnnotateMulDepth> {
  using AnnotateMulDepthBase::AnnotateMulDepthBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<MulDepthAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    walkValues(getOperation(), [&](Value value) {
      auto* lattice = solver.lookupState<MulDepthLattice>(value);
      LLVM_DEBUG(llvm::dbgs() << "lattice for value " << value << " : ");
      if (!lattice) {
        LLVM_DEBUG(llvm::dbgs() << "mul depth lattice undefined\n");
        return;
      }
      auto& state = lattice->getValue();
      if (!state.isInitialized()) {
        LLVM_DEBUG(llvm::dbgs() << "mul depth lattice uninitialized\n");
        return;
      }
      OpBuilder b(value.getContext());
      setAttributeAssociatedWith(value, "secret.mul_depth",
                                 b.getIndexAttr(state.getMulDepth()));
    });
  }
};

}  // namespace heir
}  // namespace mlir
