#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_

#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

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

template <typename Op>
struct RemoveOp : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  RemoveOp(MLIRContext *context)
      : OpRewritePattern<Op>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override;

 private:
  Operation *top;
};

// when reached a certain depth (water line), bootstrap
template <typename Op>
struct BootstrapWaterLine : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  BootstrapWaterLine(MLIRContext *context, Operation *top,
                     DataFlowSolver *solver, int waterline)
      : OpRewritePattern<Op>(context, /*benefit=*/1),
        top(top),
        solver(solver),
        waterline(waterline) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override;

 private:
  Operation *top;
  DataFlowSolver *solver;
  int waterline;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_
