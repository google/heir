#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "convert-if-to-select"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_CONVERTIFTOSELECT
#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h.inc"

struct IfToSelectConversion : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

 public:
  IfToSelectConversion(Operation* top, DataFlowSolver* solver,
                       MLIRContext* context)
      : OpRewritePattern(context), top(top), solver(solver) {}

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter& rewriter) const override {
    auto* lattice = solver->lookupState<SecretnessLattice>(ifOp.getOperand());
    Secretness secretness = lattice ? lattice->getValue() : Secretness();

    // Convert ops with "secret" and, conservatively, "unknown" (uninitialized)
    // conditions but skip conversion if the condition is known to be non-secret
    if (secretness.isInitialized() && !secretness.getSecretness())
      return failure();

    // Hoist instructions in the 'then' and 'else' regions
    auto thenOps = ifOp.getThenRegion().getOps();
    auto elseOps = ifOp.getElseRegion().getOps();

    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    for (auto& operation : llvm::make_early_inc_range(
             llvm::concat<Operation>(thenOps, elseOps))) {
      if (!isPure(&operation)) {
        // Using custom error message, as default looks bad with multiple ops
        InFlightDiagnostic diag = mlir::emitError(
            operation.getLoc(),
            "Cannot convert scf.if to arith.select, "
            "as it contains code that cannot be safely hoisted:");
        if (getContext()->shouldPrintOpOnDiagnostic()) {
          diag.attachNote(ifOp->getLoc())
              .append("containing scf.if operation:")
              .appendOp(*ifOp, OpPrintingFlags().printGenericOpForm());
        }
        return failure();
      }
      if (!llvm::isa<scf::YieldOp>(operation)) {
        rewriter.moveOpBefore(&operation, ifOp);
      }
    }

    // Translate YieldOp into SelectOp
    auto cond = ifOp.getCondition();
    auto thenYieldArgs = ifOp.thenYield().getOperands();
    auto elseYieldArgs = ifOp.elseYield().getOperands();

    SmallVector<Value> newResults(ifOp->getNumResults());
    if (ifOp->getNumResults() > 0) {
      rewriter.setInsertionPoint(ifOp);

      for (const auto& it :
           llvm::enumerate(llvm::zip(thenYieldArgs, elseYieldArgs))) {
        Value trueVal = std::get<0>(it.value());
        Value falseVal = std::get<1>(it.value());
        newResults[it.index()] = arith::SelectOp::create(
            rewriter, ifOp.getLoc(), cond, trueVal, falseVal);
      }

      rewriter.replaceOp(ifOp, newResults);
    }

    // re-run the analysis propagate secretness across
    // newly created Values
    return solver->initializeAndRun(top);
  }

 private:
  // root operation the pass is on, should never be altered hence never null
  Operation* top;
  DataFlowSolver* solver;
};

struct ConvertIfToSelect : impl::ConvertIfToSelectBase<ConvertIfToSelect> {
  using ConvertIfToSelectBase::ConvertIfToSelectBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();

    RewritePatternSet patterns(context);

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    patterns.add<IfToSelectConversion>(getOperation(), &solver, context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));

    LLVM_DEBUG({ annotateSecretness(getOperation(), &solver, true); });
  }
};

}  // namespace heir
}  // namespace mlir
