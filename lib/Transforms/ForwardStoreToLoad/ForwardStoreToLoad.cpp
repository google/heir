#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"

#include <utility>

#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "forward-store-to-load"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FORWARDSTORETOLOAD
#include "lib/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h.inc"

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<affine::AffineStoreOp> {
 public:
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap = affine::expandAffineMap(rewriter, op.getLoc(),
                                                    op.getAffineMap(), indices);
    if (!maybeExpandedMap) return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<affine::AffineLoadOp> {
 public:
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands = affine::expandAffineMap(rewriter, op.getLoc(),
                                                  op.getAffineMap(), indices);
    if (!resultOperands) return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

bool ForwardSingleStoreToLoad::isForwardableOp(Operation *potentialStore,
                                               memref::LoadOp &loadOp) const {
  if (!dominanceInfo.properlyDominates(potentialStore, loadOp.getOperation())) {
    LLVM_DEBUG(llvm::dbgs() << "store op does not dominate load op\n");
    return false;
  }

  // Probably want to relax this at some point in the future.
  if (loadOp->getBlock() != potentialStore->getBlock()) {
    LLVM_DEBUG(llvm::dbgs()
               << "loadOp and store op are not in the same block\n");
    return false;
  }

  return llvm::TypeSwitch<Operation &, bool>(*potentialStore)
      .Case<memref::StoreOp>([&](auto storeOp) {
        ValueRange storeIndices = storeOp.getIndices();
        ValueRange loadIndices = loadOp.getIndices();
        if (storeIndices != loadIndices) {
          LLVM_DEBUG(llvm::dbgs()
                     << "loadOp and store op do not have matching indices\n");
          return false;
        }

        // Naively scan through the operations between the two ops and check if
        // anything prevents forwarding.
        for (auto currentNode = storeOp->getNextNode();
             currentNode != loadOp.getOperation();
             currentNode = currentNode->getNextNode()) {
          if (currentNode->getNumRegions() > 0) {
            // Op can have control flow
            LLVM_DEBUG(llvm::dbgs() << "an op with control flow is between the "
                                       "store and load op\n");
            return false;
          }
          if (auto op = dyn_cast<memref::StoreOp>(currentNode)) {
            if (op.getMemRef() == storeOp.getMemRef() &&
                op.getIndices() == storeIndices) {
              LLVM_DEBUG(llvm::dbgs()
                         << "an intermediate op stores to the same index\n");
              return false;
            }
          }
        }

        return true;
      })
      .Default([&](Operation &) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unsupported op type, cannot check for forwardability\n");
        return false;
      });
}

FailureOr<Value> getStoredValue(Operation *storeOp) {
  return llvm::TypeSwitch<Operation &, FailureOr<Value>>(*storeOp)
      .Case<memref::StoreOp>(
          [&](auto storeOp) { return storeOp.getValueToStore(); })
      .Default([&](Operation &) { return failure(); });
}

LogicalResult ForwardSingleStoreToLoad::matchAndRewrite(
    memref::LoadOp loadOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "Considering loadOp for replacement: " << loadOp
                          << "\n");
  for (Operation *use : loadOp.getMemRef().getUsers()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Considering memref use for forwarding: " << *use << "\n");
    if (isForwardableOp(use, loadOp)) {
      auto result = getStoredValue(use);
      LLVM_DEBUG(llvm::dbgs() << "Use is forwardable: " << *use << "\n");
      if (failed(result)) {
        return failure();
      }
      auto value = result.value();
      rewriter.replaceAllUsesWith(loadOp, value);
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "Use is not forwardable: " << *use << "\n");
  }
  return failure();
}

bool RemoveUnusedStore::isPostDominated(Operation *potentialOp,
                                        memref::StoreOp &storeOp) const {
  if (!dominanceInfo.properlyDominates(storeOp.getOperation(), potentialOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "store op is not properly dominated by potential store\n");
    return false;
  }

  // Probably want to relax this at some point in the future.
  if (storeOp->getBlock() != potentialOp->getBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "store ops are not in the same block\n");
    return false;
  }

  return llvm::TypeSwitch<Operation &, bool>(*potentialOp)
      .Case<memref::StoreOp>([&](auto potentialStore) {
        ValueRange storeIndices = storeOp.getIndices();
        ValueRange potentialStoreIndices = potentialStore.getIndices();
        if (storeIndices != potentialStoreIndices) {
          LLVM_DEBUG(llvm::dbgs()
                     << "store ops do not have matching indices\n");
          return false;
        }

        // Naively scan through the operations between the two ops and check if
        // a read prevents store removal.
        for (auto currentNode = storeOp->getNextNode();
             currentNode != potentialStore.getOperation();
             currentNode = currentNode->getNextNode()) {
          if (currentNode->getNumRegions() > 0) {
            // Op can have control flow
            LLVM_DEBUG(llvm::dbgs() << "an op with control flow is between the "
                                       "store ops\n");
            return false;
          }
          if (auto op = dyn_cast<affine::AffineLoadOp>(currentNode)) {
            // If we encounter an affine load op then fail conservatively. This
            // pass should have already run AffineLoadLowering to convert all
            // possible affine loads to memref loads.
            LLVM_DEBUG(llvm::dbgs() << "an intermediate load op was found\n");
            return false;
          }
          if (auto op = dyn_cast<memref::LoadOp>(currentNode)) {
            if (op.getMemRef() == storeOp.getMemRef() &&
                op.getIndices() == storeIndices) {
              LLVM_DEBUG(llvm::dbgs()
                         << "an intermediate op loads at the same index\n");
              return false;
            }
          }
        }

        return true;
      })
      .Default([&](Operation &) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unsupported op type, cannot check for forwardability\n");
        return false;
      });
}

LogicalResult RemoveUnusedStore::matchAndRewrite(
    memref::StoreOp storeOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "Considering storeOp for removal: " << storeOp
                          << "\n");
  for (Operation *use : storeOp.getMemRef().getUsers()) {
    if (isPostDominated(use, storeOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Store is usurped by: " << *use << "\n");
      rewriter.eraseOp(storeOp);
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "Use is not removable: " << *use << "\n");
  }
  return failure();
}

struct ForwardStoreToLoad : impl::ForwardStoreToLoadBase<ForwardStoreToLoad> {
  using ForwardStoreToLoadBase::ForwardStoreToLoadBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    DominanceInfo dom(getOperation());
    patterns.add<AffineLoadLowering, AffineStoreLowering>(context);
    patterns.add<ForwardSingleStoreToLoad, RemoveUnusedStore>(context, dom);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
