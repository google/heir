#include "lib/Transforms/ConvertSecretWhileToStaticFor/ConvertSecretWhileToStaticFor.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_CONVERTSECRETWHILETOSTATICFOR
#include "lib/Transforms/ConvertSecretWhileToStaticFor/ConvertSecretWhileToStaticFor.h.inc"

struct SecretWhileToStaticForConversion : OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

 public:
  SecretWhileToStaticForConversion(DataFlowSolver *solver, MLIRContext *context)
      : OpRewritePattern(context), solver(solver) {}

  // TODO(#846): Add support for do-while loops
  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    auto conditionOp = whileOp.getConditionOp();
    auto whileCondition = conditionOp->getOperand(0);

    auto *beforeRegionArgs = whileOp.getBefore().front().getArguments().data();

    auto conditionOpArgs =
        // Get Loop variables: includes all arguments without the condition
        conditionOp->getOperands().drop_front(1);

    int counter = 0;
    for (auto arg : conditionOpArgs) {
      // Check if the defining operation of the loop variable passed in
      // scf.condition is the same as the operation defining the block argument
      // passed by scf.while. A true value implies that the loop variable is the
      // same as the block argument, i.e., the original block argument passed by
      // scf.while has not been changed.
      bool isEqual =
          arg.getDefiningOp() == beforeRegionArgs[counter].getDefiningOp();
      if (!isEqual) {
        InFlightDiagnostic diag =
            whileOp.emitWarning()
            << "Current loop transformation has no support for do-while loops:";
        return failure();
      }
      counter++;
    }

    auto *conditionSecretnessLattice =
        solver->lookupState<SecretnessLattice>(whileCondition);

    if (!conditionSecretnessLattice) {
      InFlightDiagnostic diag =
          whileOp.emitWarning()
          << "Secretness for scf.while condition has not been set";
      return failure();
    }

    bool isConditionSecret =
        conditionSecretnessLattice->getValue().getSecretness();

    // If condition is not secret, no transformation is needed
    if (!isConditionSecret) {
      return failure();
    }

    auto maxIterAttr = whileOp->getAttrOfType<IntegerAttr>("max_iter");

    // If maximum iteration attribute is not provided, emit a warning and return
    // failure
    if (!maxIterAttr) {
      whileOp.emitWarning()
          << "Cannot convert secret scf.while to static affine.for "
             "since a static maximum iteration attribute (`max_iter`) has not "
             "been provided on the scf.while op:";
      return failure();
    }

    ImplicitLocOpBuilder builder(whileOp->getLoc(), rewriter);

    int upperBound = maxIterAttr.getInt();

    SmallVector<Value> iterArgs(whileOp.getInits());

    auto newForOp =
        builder.create<affine::AffineForOp>(0, upperBound, 1, iterArgs);
    newForOp->setAttrs(whileOp->getAttrs());

    builder.setInsertionPointToStart(newForOp.getBody());

    IRMapping mp;
    for (BlockArgument blockArg : whileOp.getBody()->getArguments()) {
      mp.map(blockArg,
             /* Adding 1 to bypass the first block argument of newForOp, i.e.,
                the induction variable  */
             newForOp.getBody()->getArguments()[blockArg.getArgNumber() + 1]);
    }

    Value condition;
    for (auto &op : whileOp.getBefore().getOps()) {
      if (auto conditionOp = dyn_cast<scf::ConditionOp>(op)) {
        // Get condition used in scf.condition op
        condition = mp.lookup(conditionOp.getOperand(0));
      } else {
        // Copy op in "before" region to the forOp's body
        builder.clone(op, mp);
      }
    }

    auto ifOp = builder.create<scf::IfOp>(
        condition,
        [&](OpBuilder &b, Location loc) {
          for (BlockArgument blockArg : whileOp.getAfterArguments()) {
            mp.map(blockArg, newForOp.getBody()
                                 ->getArguments()[blockArg.getArgNumber() + 1]);
          }
          for (auto &op : whileOp.getAfter().getOps()) {
            // Copy op in "after" region to the ifOp's "then" region
            b.clone(op, mp);
          }
        },
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, newForOp.getRegionIterArgs());
        });

    // Create YieldOp for affine.for
    builder.create<affine::AffineYieldOp>(ifOp.getResults());

    // Replace scf.while with affine.for
    rewriter.replaceOp(whileOp, newForOp);

    return success();
  }

 private:
  DataFlowSolver *solver;
};

struct ConvertSecretWhileToStaticFor
    : impl::ConvertSecretWhileToStaticForBase<ConvertSecretWhileToStaticFor> {
  using ConvertSecretWhileToStaticForBase::ConvertSecretWhileToStaticForBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    patterns.add<SecretWhileToStaticForConversion>(&solver, context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
