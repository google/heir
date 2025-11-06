#include "lib/Transforms/UnusedMemRef/UnusedMemRef.h"

#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_REMOVEUNUSEDMEMREF
#include "lib/Transforms/UnusedMemRef/UnusedMemRef.h.inc"

class RemoveUnusedMemrefPattern : public OpRewritePattern<memref::AllocOp> {
 public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter& rewriter) const override {
    auto memref = op.getMemref();
    if (llvm::any_of(memref.getUsers(), [&](Operation* ownerOp) {
          return !isa<affine::AffineWriteOpInterface>(ownerOp) &&
                 !isa<memref::StoreOp>(ownerOp) &&
                 !hasSingleEffect<MemoryEffects::Free>(ownerOp, memref);
        })) {
      return rewriter.notifyMatchFailure(
          op,
          "memref has users that are not affine write ops, memref store "
          "ops, or do not have a single MemoryEffects::Free effect");
    }
    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto* user : llvm::make_early_inc_range(memref.getUsers())) {
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct RemoveUnusedMemRef
    : public impl::RemoveUnusedMemRefBase<RemoveUnusedMemRef> {
  using RemoveUnusedMemRefBase::RemoveUnusedMemRefBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveUnusedMemrefPattern>(context);
    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
