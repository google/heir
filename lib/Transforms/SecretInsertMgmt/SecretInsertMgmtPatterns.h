#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_

#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project

namespace mlir {
namespace heir {

template <typename MulOp>
struct MultRelinearize : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  MultRelinearize(MLIRContext *context, Operation *top, DataFlowSolver *solver)
      : OpRewritePattern<MulOp>(context, /*benefit=*/1),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(MulOp mulOp,
                                PatternRewriter &rewriter) const override;

 private:
  Operation *top;
  DataFlowSolver *solver;
};

template <typename Op>
struct ModReduceBefore : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  ModReduceBefore(MLIRContext *context, bool isMul, bool includeFirstMul,
                  Operation *top, DataFlowSolver *solver)
      : OpRewritePattern<Op>(context, /*benefit=*/1),
        isMul(isMul),
        includeFirstMul(includeFirstMul),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override;

 private:
  bool isMul;
  bool includeFirstMul;
  Operation *top;
  DataFlowSolver *solver;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_
