#ifndef LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_
#define LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_

#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

/// Insert mgmt.relin op immediately after the matched mul op.
template <typename MulOp>
struct MultRelinearize : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  MultRelinearize(MLIRContext* context, Operation* top, DataFlowSolver* solver)
      : OpRewritePattern<MulOp>(context, /*benefit=*/1),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(MulOp mulOp,
                                PatternRewriter& rewriter) const override;

 private:
  Operation* top;
  DataFlowSolver* solver;
};

/// Insert mgmt.modreduce op immediately after the matched mul op.
template <typename MulOp>
struct ModReduceAfterMult : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  ModReduceAfterMult(MLIRContext* context, Operation* top,
                     DataFlowSolver* solver)
      : OpRewritePattern<MulOp>(context, /*benefit=*/1),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override;

 private:
  Operation* top;
  DataFlowSolver* solver;
};

/// Insert mgmt.modreduce op for each operand of the matched mul op.
template <typename Op>
struct ModReduceBefore : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  ModReduceBefore(MLIRContext* context, bool includeFirstMul, Operation* top,
                  DataFlowSolver* solver)
      : OpRewritePattern<Op>(context, /*benefit=*/1),
        includeFirstMul(includeFirstMul),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter& rewriter) const override;

 private:
  bool includeFirstMul;
  Operation* top;
  DataFlowSolver* solver;
};

/// Insert cross level operation sequence to match the level of operands
///
/// (optional level_reduce) + adjust_scale + mod_reduce
///
/// Note its interaction with ModReduceBefore: it will insert mod reduce
/// blindly for mul op, and it becomes mul(mr(op0), mr(op1)). Then this pattern
/// will find that rs(op0) and rs(op1) are not at the same level, and insert
/// (level reduce) + adjust scale + mod reduce, then one of them will
/// become mod reduce + (level reduce) + adjust scale + mod reduce. Then
/// we rely on canonicalization to re-order and merge them into standard
/// sequence (level reduce) + adjust scale + mod reduce.
///
/// When the implementation of relevant patterns changed, please make sure
/// the canonicalization patterns are updated accordingly.
///
/// The behavior is described also in the Ciphertext management section of the
/// document.
template <typename Op>
struct MatchCrossLevel : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  MatchCrossLevel(MLIRContext* context, int* idCounter, Operation* top,
                  DataFlowSolver* solver)
      : OpRewritePattern<Op>(context, /*benefit=*/1),
        idCounter(idCounter),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter& rewriter) const override;

 private:
  int* idCounter;
  Operation* top;
  DataFlowSolver* solver;
};

/// Similar to MatchCrossLevel, see its description for behavior.
///
/// For inserting-modulus-switching-before-mul (not include-first-mul case),
/// for add(res = mul(input0, input1), input2), res and input2 may not have
/// the same scale, yet they are at the same level, so we need to adjust the
/// scale of input2 to match the scale of res.
///
/// Note that this only happens at the first level, as in other levels this
/// mismatch will be resolved by MatchCrossLevel.
template <typename Op>
struct MatchCrossMulDepth : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  MatchCrossMulDepth(MLIRContext* context, int* idCounter, Operation* top,
                     DataFlowSolver* solver)
      : OpRewritePattern<Op>(context, /*benefit=*/1),
        idCounter(idCounter),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter& rewriter) const override;

 private:
  int* idCounter;
  Operation* top;
  DataFlowSolver* solver;
};

/// Insert mgmt.init op for plaintext operand.
///
/// See the documentation for mgmt.init for more details.
template <typename Op>
struct UseInitOpForPlaintextOperand : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  UseInitOpForPlaintextOperand(MLIRContext* context, Operation* top,
                               DataFlowSolver* solver)
      : OpRewritePattern<Op>(context, /*benefit=*/1),
        top(top),
        solver(solver) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter& rewriter) const override;

 private:
  Operation* top;
  DataFlowSolver* solver;
};

/// when reached a certain depth (water line), bootstrap
///
/// TODO(#1642): make it work with cross-level operation
template <typename Op>
struct BootstrapWaterLine : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  BootstrapWaterLine(MLIRContext* context, Operation* top,
                     DataFlowSolver* solver, int waterline)
      : OpRewritePattern<Op>(context, /*benefit=*/1),
        top(top),
        solver(solver),
        waterline(waterline) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter& rewriter) const override;

 private:
  Operation* top;
  DataFlowSolver* solver;
  int waterline;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SECRETINSERTMGMT_SECRETINSERTMGMTPATTERNS_H_
