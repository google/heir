#include "include/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h"

#include <utility>

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"              // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "forward-store-to-load"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FORWARDSTORETOLOAD
#include "include/Transforms/ForwardStoreToLoad/ForwardStoreToLoad.h.inc"

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
        // get this node to the load node and check if any in between
        // isForwardableOp

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

struct ForwardStoreToLoad : impl::ForwardStoreToLoadBase<ForwardStoreToLoad> {
  using ForwardStoreToLoadBase::ForwardStoreToLoadBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    DominanceInfo dom(getOperation());
    patterns.add<ForwardSingleStoreToLoad>(context, dom);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
