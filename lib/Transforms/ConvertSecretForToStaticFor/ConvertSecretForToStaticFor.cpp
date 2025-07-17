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
  SecretForToStaticForConversion(DataFlowSolver *solver, MLIRContext *context)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    bool isLowerBoundSecret = isSecret(forOp.getLowerBound(), solver);
    bool isUpperBoundSecret = isSecret(forOp.getUpperBound(), solver);

    // If both bounds are non-secret constants, return
    if (!isLowerBoundSecret && !isUpperBoundSecret) return failure();

    int newLowerBound, newUpperBound;

    if (isLowerBoundSecret) {
      // If static lower bound is not provided, emit a warning and return
      // failure
      if (!forOp->getAttrOfType<IntegerAttr>("lower")) {
        InFlightDiagnostic diag = mlir::emitWarning(
            forOp.getLoc(),
            "Cannot convert secret scf.for to static affine.for "
            "since a static lower bound attribute has not been provided:");

        return failure();
      }
      // If lowerBound is secret, get value from "lower" attribute
      newLowerBound = forOp->getAttrOfType<IntegerAttr>("lower").getInt();
    }

    if (isUpperBoundSecret) {
      // If static upper bound is not provided, emit a warning and return
      // failure
      if (!forOp->getAttrOfType<IntegerAttr>("upper")) {
        InFlightDiagnostic diag = mlir::emitWarning(
            forOp.getLoc(),
            "Cannot convert secret scf.for to static affine.for "
            "since a static upper bound attribute has not been provided:");

        return failure();
      }
      // If upperBound is secret, get value from "upper" attribute
      newUpperBound = forOp->getAttrOfType<IntegerAttr>("upper").getInt();
    }

    ImplicitLocOpBuilder builder(forOp->getLoc(), rewriter);

    auto newForOp = affine::AffineForOp::create(
        builder,
        isLowerBoundSecret ? newLowerBound
                           : forOp.getLowerBound()
                                 .getDefiningOp()
                                 ->getAttrOfType<IntegerAttr>("value")
                                 .getInt(),
        isUpperBoundSecret ? newUpperBound
                           : forOp.getUpperBound()
                                 .getDefiningOp()
                                 ->getAttrOfType<IntegerAttr>("value")
                                 .getInt(),
        forOp.getStep()
            .getDefiningOp()
            ->getAttrOfType<IntegerAttr>("value")
            .getInt(),
        forOp.getInitArgs());

    newForOp->setAttrs(forOp->getAttrs());

    auto inductionVariable = newForOp.getInductionVar();

    builder.setInsertionPointToStart(newForOp.getBody());

    arith::CmpIOp cmpIUpper, cmpILower;
    arith::AndIOp andI;

    if (isLowerBoundSecret) {
      // Create arith.cmpi (iv >= oldLowerBound)
      cmpILower =
          arith::CmpIOp::create(builder, arith::CmpIPredicate::sge,
                                inductionVariable, forOp.getLowerBound());
    }

    if (isUpperBoundSecret) {
      // Create arith.cmpi (iv < oldUpperBound)
      cmpIUpper =
          arith::CmpIOp::create(builder, arith::CmpIPredicate::slt,
                                inductionVariable, forOp.getUpperBound());
    }

    // If both lowerBound and upperBound are secret, join the two arith.cmpi
    // with an arith.andi
    if (isLowerBoundSecret && isUpperBoundSecret) {
      andI = arith::AndIOp::create(builder, cmpILower, cmpIUpper);
    }

    // Create scf.if to conditionally execute the body of scf.for
    auto cond = isLowerBoundSecret ? isUpperBoundSecret ? andI.getResult()
                                                        : cmpILower.getResult()
                                   : cmpIUpper.getResult();
    scf::IfOp ifOp = scf::IfOp::create(
        builder, cond,
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
          scf::YieldOp::create(b, loc, newForOp.getRegionIterArgs());
        });

    // Create YieldOp for affine.for
    affine::AffineYieldOp::create(builder, ifOp.getResults());

    // Replace scf.for with affine.for
    rewriter.replaceOp(forOp, newForOp);

    return success();
  }

 private:
  DataFlowSolver *solver;
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
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    patterns.add<SecretForToStaticForConversion>(&solver, context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
