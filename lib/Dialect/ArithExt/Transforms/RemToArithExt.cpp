#include "lib/Dialect/ArithExt/Transforms/RemToArithExt.h"

#include "lib/Dialect/ArithExt/IR/ArithExtOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"

#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace arith_ext {

#define GEN_PASS_DEF_REMTOARITHEXT
#include "lib/Dialect/ArithExt/Transforms/Passes.h.inc"

struct RemToArithExt : impl::RemToArithExtBase<RemToArithExt> {
  using RemToArithExtBase::RemToArithExtBase;

  void runOnOperation() override {
    Operation *module = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(module)))
        signalPassFailure();

    auto result = module->walk([&](Operation *op) {
      if (!llvm::isa<arith::RemUIOp>(*op)) {
        return WalkResult::advance();
      }

      const dataflow::IntegerValueRangeLattice *opRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(op->getResult(0));
      if (!opRange || opRange->getValue().isUninitialized()) {
        op->emitOpError()
          << "No op range was given.";
        return WalkResult::interrupt();
      }
      opRange->print(llvm::errs());
      return WalkResult::advance();
    });
  }
};

}  // namespace arith_ext
}  // namespace heir
}  // namespace mlir
