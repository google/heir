#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h"

#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"      // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_CONVERTIFTOSELECT
#include "lib/Transforms/ConvertIfToSelect/ConvertIfToSelect.h.inc"

struct IfToSelectConversion : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Hoist instructions in the 'then' and 'else' regions
    auto thenOps = ifOp.getThenRegion().getOps();
    auto elseOps = ifOp.getElseRegion().getOps();

    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    for (auto &operation : llvm::make_early_inc_range(
             llvm::concat<Operation>(thenOps, elseOps))) {
      if (!isPure(&operation)) {
        ifOp->emitError()
            << "Can't convert scf.if to arith.select operation. If-operation "
               "contains code that can't be safely hoisted on line "
            << operation.getLoc();
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

    SmallVector<Value> newIfResults(ifOp->getNumResults());
    if (ifOp->getNumResults() > 0) {
      rewriter.setInsertionPoint(ifOp);

      for (const auto &it :
           llvm::enumerate(llvm::zip(thenYieldArgs, elseYieldArgs))) {
        Value trueVal = std::get<0>(it.value());
        Value falseVal = std::get<1>(it.value());
        newIfResults[it.index()] = rewriter.create<arith::SelectOp>(
            ifOp.getLoc(), cond, trueVal, falseVal);
      }
      rewriter.replaceOp(ifOp, newIfResults);
    }

    return success();
  }
};

struct ConvertIfToSelect : impl::ConvertIfToSelectBase<ConvertIfToSelect> {
  using ConvertIfToSelectBase::ConvertIfToSelectBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<IfToSelectConversion>(context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
