#include "lib/Transforms/ConvertSecretForToStaticFor/ConvertSecretForToStaticFor.h"

#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
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

#define GEN_PASS_DEF_CONVERTSECRETFORTOSTATICFOR
#include "lib/Transforms/ConvertSecretForToStaticFor/ConvertSecretForToStaticFor.h.inc"

struct SecretForToStaticForConversion : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

 public:
  SecretForToStaticForConversion(DataFlowSolver *solver, MLIRContext *context,
                                 bool convertAllScfFor)
      : OpRewritePattern(context),
        solver(solver),
        convertAllScfFor(convertAllScfFor) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();

    bool isLowerBoundSecret = isSecret(lowerBound, solver);
    bool isUpperBoundSecret = isSecret(upperBound, solver);

    // If both bounds are non-secret constants and we're not converting all
    // scf.for ops, do nothing
    if (!convertAllScfFor && !isLowerBoundSecret && !isUpperBoundSecret)
      return failure();

    // Affine.for can only handle strict static (integer attribute) bounds,
    // but scf.for can handle dynamic bounds, so we need to check:
    if (!forOp.getConstantStep())
      return emitError(forOp.getLoc(),
                       "Cannot convert secret scf.for to static affine.for "
                       "since the step is not constant and "
                       "affine.for only supports strictly static step size");

    // There are a few cases to handle:
    // If a bound is determined to be secret, we need to replace it
    // with the annotated static bound, which we wrap in a constant op
    // for simplicity (this should later be optimized away into the affine map).
    //
    // Even the bound is not secret, we might need to cast it to IndexType
    // since affine.for (in contrast to scf.for)
    // does not allow signless integers for the bounds
    //
    // Finally, once we have two suitable values, we need to construct an affine
    // map (since the non-secret bound might still be dynamic). Thankfully,
    // scf.for only allows a simple start, end, step form, so we can create a
    // relatively simple affine map

    ImplicitLocOpBuilder builder(forOp->getLoc(), rewriter);

    if (!isa<IndexType>(lowerBound.getType())) {
      lowerBound = builder.create<arith::IndexCastOp>(
          lowerBound.getLoc(), builder.getIndexType(), lowerBound);
    }
    if (!isa<IndexType>(upperBound.getType())) {
      upperBound = builder.create<arith::IndexCastOp>(
          upperBound.getLoc(), builder.getIndexType(), upperBound);
    }
    Value newLowerBound = lowerBound;
    Value newUpperBound = upperBound;

    if (isLowerBoundSecret) {
      // If static lower bound is not provided, emit an error and return
      if (auto lowerBoundAttr = forOp->getAttrOfType<IntegerAttr>("lower")) {
        newLowerBound = builder.create<arith::ConstantIndexOp>(
            lowerBound.getLoc(), lowerBoundAttr.getInt());
      } else {
        return emitError(
            forOp.getLoc(),
            "Cannot convert secret scf.for to static affine.for "
            "since a static lower bound attribute has not been provided:");
      }
    }

    if (isUpperBoundSecret) {
      // If static upper bound is not provided, emit an error and return
      if (auto upperBoundAttr = forOp->getAttrOfType<IntegerAttr>("upper")) {
        newUpperBound = builder.create<arith::ConstantIndexOp>(
            upperBound.getLoc(), upperBoundAttr.getInt());
      } else {
        return emitError(
            forOp.getLoc(),
            "Cannot convert secret scf.for to static affine.for "
            "since a static upper bound attribute has not been provided:");
      }
    }

    auto newForOp = builder.create<affine::AffineForOp>(
        ValueRange(newLowerBound), builder.getSymbolIdentityMap(),
        ValueRange(newUpperBound), builder.getSymbolIdentityMap(),
        forOp.getConstantStep()->getLimitedValue(), forOp.getInitArgs());

    newForOp->setAttrs(forOp->getAttrs());

    auto inductionVariable = newForOp.getInductionVar();

    builder.setInsertionPointToStart(newForOp.getBody());

    if (!isLowerBoundSecret && !isUpperBoundSecret) {
      // If neither bound is secret,
      // we can directly copy the body
      IRMapping mp;
      for (BlockArgument blockArg : forOp.getBody()->getArguments()) {
        mp.map(blockArg,
               newForOp.getBody()->getArguments()[blockArg.getArgNumber()]);
      }
      for (auto &op : forOp.getBody()->getOperations()) {
        // Convert scf.yield to affine.yield
        if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
          SmallVector<Value> mappedOperands;
          for (Value operand : yieldOp.getOperands()) {
            mappedOperands.push_back(mp.lookupOrDefault(operand));
          }
          builder.create<affine::AffineYieldOp>(yieldOp.getLoc(),
                                                mappedOperands);
        } else {
          builder.clone(op, mp);
        }
      }
      // Replace scf.for with affine.for
      rewriter.replaceOp(forOp, newForOp);
      return success();
    }

    // Handle secret bounds with conditional logic
    arith::CmpIOp cmpIUpper, cmpILower;
    arith::AndIOp andI;

    if (isLowerBoundSecret) {
      // Create arith.cmpi (iv >= oldLowerBound)
      cmpILower = builder.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                                inductionVariable, lowerBound);
    }

    if (isUpperBoundSecret) {
      // Create arith.cmpi (iv < oldUpperBound)
      cmpIUpper = builder.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                inductionVariable, upperBound);
    }

    // If both lowerBound and upperBound are secret, join the two arith.cmpi
    // with an arith.andi
    if (isLowerBoundSecret && isUpperBoundSecret) {
      andI = builder.create<arith::AndIOp>(cmpILower, cmpIUpper);
    }

    // Create scf.if to conditionally execute the body of scf.for
    auto cond = isLowerBoundSecret ? isUpperBoundSecret ? andI.getResult()
                                                        : cmpILower.getResult()
                                   : cmpIUpper.getResult();
    scf::IfOp ifOp = builder.create<scf::IfOp>(
        cond,
        // 'Then' region
        [&](OpBuilder &b, Location loc) {
          // Copy body of the scf::ForOp
          IRMapping mp;
          for (BlockArgument blockArg : forOp.getBody()->getArguments()) {
            mp.map(blockArg,
                   newForOp.getBody()->getArguments()[blockArg.getArgNumber()]);
          }
          for (auto &op : forOp.getBody()->getOperations()) {
            b.clone(op, mp);
          }
        },
        // 'Else' region
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, newForOp.getRegionIterArgs());
        });

    // Create YieldOp for affine.for
    builder.create<affine::AffineYieldOp>(ifOp.getResults());

    // Replace scf.for with affine.for
    rewriter.replaceOp(forOp, newForOp);

    return success();
  }

 private:
  DataFlowSolver *solver;
  bool convertAllScfFor;
};

struct ConvertSecretForToStaticFor
    : impl::ConvertSecretForToStaticForBase<ConvertSecretForToStaticFor> {
  using ConvertSecretForToStaticForBase::ConvertSecretForToStaticForBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(getOperation());

    if (failed(result)) {
      emitError(getOperation()->getLoc(), "Failed to run the analysis.\n");
      signalPassFailure();
      return;
    }

    patterns.add<SecretForToStaticForConversion>(&solver, context,
                                                 this->convertAllScfFor);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
