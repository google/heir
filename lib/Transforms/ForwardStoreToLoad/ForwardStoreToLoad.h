#ifndef LIB_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_H_
#define LIB_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_H_

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h.inc"

// Find a memref load and try to forward the most recent store op.
struct ForwardSingleStoreToLoad : public OpRewritePattern<memref::LoadOp> {
  ForwardSingleStoreToLoad(mlir::MLIRContext *context, DominanceInfo &dom)
      : OpRewritePattern<memref::LoadOp>(context, /*benefit=*/3),
        dominanceInfo(dom) {}

 public:
  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override;

 private:
  // Updates an internal cache with results of this query so they can be used
  // recursively.
  bool isForwardableOp(Operation *potentialStore, memref::LoadOp &loadOp) const;

  DominanceInfo &dominanceInfo;
};

// Perform unused store elimination
struct RemoveUnusedStore : public OpRewritePattern<memref::StoreOp> {
  RemoveUnusedStore(mlir::MLIRContext *context, DominanceInfo &dom)
      : OpRewritePattern<memref::StoreOp>(context, /*benefit=*/3),
        dominanceInfo(dom) {}

 public:
  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override;

 private:
  bool isPostDominated(Operation *potentialOp, memref::StoreOp &storeOp) const;

  DominanceInfo &dominanceInfo;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_H_
