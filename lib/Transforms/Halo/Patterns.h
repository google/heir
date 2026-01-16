#ifndef LIB_TRANSFORMS_HALO_PATTERNS_H_
#define LIB_TRANSFORMS_HALO_PATTERNS_H_

#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project

namespace mlir {
namespace heir {

struct PeelPlaintextAffineForInit
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  PeelPlaintextAffineForInit(MLIRContext* context, DataFlowSolver* solver)
      : OpRewritePattern(context), solver(solver) {}

 public:
  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                PatternRewriter& rewriter) const override;

 private:
  DataFlowSolver* solver;
};

struct PeelPlaintextScfForInit : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  PeelPlaintextScfForInit(MLIRContext* context, DataFlowSolver* solver)
      : OpRewritePattern(context), solver(solver) {}

 public:
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter& rewriter) const override;

 private:
  DataFlowSolver* solver;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_HALO_PATTERNS_H_
