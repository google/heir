#include <iostream>

#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "include/Conversion/MemrefToArith/Utils.h"
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h" // from @llvm-project
#include "mlir/include/mlir/IR/Location.h" // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/include/mlir/IR/Types.h" // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h" // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h" // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h" // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h" // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h" // from @llvm-project

namespace mlir {
namespace heir {

namespace {

static LogicalResult forwardFullyUnrolledStoreToLoad(
    mlir::affine::AffineReadOpInterface loadOp,
    SmallVectorImpl<Operation *> &opsToErase,
    llvm::DenseMap<std::pair<mlir::Value, uint64_t>, mlir::Operation *>
        &storeMap) {
  std::optional<mlir::Operation *> storeOpOrNull;
  auto loadMemRef = loadOp.getMemRef();
  affine::MemRefAccess dstAccess(loadOp);
  auto accessIndices = getFlattenedAccessIndex(dstAccess, loadMemRef.getType());
  if (!accessIndices) {
    return failure();
  }

  // Check if a corresponding storeOp already exists in the storeMap.
  if (storeMap.contains(std::make_pair(loadMemRef, accessIndices.value()))) {
    storeOpOrNull = storeMap[std::make_pair(loadMemRef, accessIndices.value())];
  } else {
    // Look for an AffineWriteOp in all other users of the memref.
    for (auto user : loadMemRef.getUsers()) {
      auto storeOp = dyn_cast<mlir::affine::AffineWriteOpInterface>(user);
      if (!storeOp) {
        continue;
      }

      affine::MemRefAccess srcAccess(storeOp);
      if (srcAccess != dstAccess) {
        // Insert into map for another load to use.
        auto srcIndices =
            getFlattenedAccessIndex(srcAccess, storeOp.getMemRef().getType());
        storeMap.insert(std::make_pair(
            std::make_pair(loadMemRef, srcIndices.value()), storeOp));
        continue;
      }

      // Since we forward only from the immediately prior store, we verify that
      // there are neither intervening stores nor intervening loads in between.
      if (!mlir::affine::hasNoInterveningEffect<MemoryEffects::Write>(storeOp,
                                                                      loadOp)) {
        return failure();
      }

      // We found a match!
      storeOpOrNull = storeOp;
      break;
    }
  }

  if (!storeOpOrNull.has_value()) {
    std::cout << "could not find " << std::endl;
    return failure();
  }

  // Perform the actual store to load forwarding.
  Value storeVal =
      cast<mlir::affine::AffineWriteOpInterface>(storeOpOrNull.value())
          .getValueToStore();
  // Check if 2 values have the same shape. This is needed for affine
  // vector loads and stores.
  if (storeVal.getType() != loadOp.getValue().getType()) {
    return failure();
  }
  // Replace the load value with the store value.
  loadOp.getValue().replaceAllUsesWith(storeVal);
  // Record the store and load to erase later.
  opsToErase.push_back(storeOpOrNull.value());
  opsToErase.push_back(loadOp);
  return success();
}

}  // namespace

// FIXME: Consider anchoring on outer for loops.
class UnrollAndForwardPattern final : public mlir::RewritePattern {
 public:
  explicit UnrollAndForwardPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::func::FuncOp::getOperationName(), 1,
                             context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    auto func = mlir::cast<mlir::func::FuncOp>(op);

    // Hold an intermediate result map from [Value and flat index] to storeOp.
    llvm::DenseMap<std::pair<mlir::Value, uint64_t>, mlir::Operation *>
        storeMap;

    rewriter.startRootUpdate(func);
    auto outerLoops = func.getOps<mlir::affine::AffineForOp>();
    for (auto root : llvm::make_early_inc_range(outerLoops)) {
      // Keep track of the position of the next operation after the outer for
      // loop.
      auto nextOp = root->getNextNode();

      SmallVector<mlir::affine::AffineForOp> nestedLoops;
      mlir::affine::getPerfectlyNestedLoops(nestedLoops, root);
      nestedLoops[0].getBody(0)->walk<WalkOrder::PostOrder>(
          [&](mlir::affine::AffineForOp forOp) {
            auto unrollFactor =
                mlir::affine::getConstantTripCount(forOp).value_or(
                    std::numeric_limits<int>::max());
            if (failed(loopUnrollUpToFactor(forOp, unrollFactor))) {
              return WalkResult::skip();
            }
            return WalkResult::advance();
          });

      auto unrollFactor = mlir::affine::getConstantTripCount(root).value_or(
          std::numeric_limits<int>::max());
      if (failed(loopUnrollUpToFactor(root, unrollFactor))) return failure();

      //  Walk all load's and perform store to load forwarding.
      SmallVector<Operation *, 8> opsToErase;
      func.walk<WalkOrder::PreOrder>(
          [&](mlir::affine::AffineReadOpInterface loadOp) {
            if (loadOp->getParentOp() != nextOp->getParentOp() ||
                nextOp->isBeforeInBlock(loadOp)) {
              // Only iterate on the loads we just unravelled. Because we walk
              // in pre-order, we can interrupt the walk at this point.
              return WalkResult::interrupt();
            }

            if (failed(forwardFullyUnrolledStoreToLoad(loadOp, opsToErase,
                                                       storeMap))) {
              return WalkResult::skip();
            }
            return WalkResult::advance();
          });

      // Erase all load op's whose results were replaced with store fwd'ed
      // ones.
      for (auto *op : opsToErase) {
        op->erase();
      }
      opsToErase.clear();
    }

    rewriter.finalizeRootUpdate(func);
    return mlir::success();
  };
};

// UnrollAndForwardPass intends to forward scalars.
struct UnrollAndForwardPass
    //  : public impl::UnrollAndForwardPassBase<UnrollAndForwardPass> {
    : public mlir::PassWrapper<UnrollAndForwardPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() {
    mlir::ConversionTarget target(getContext());

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<UnrollAndForwardPattern>(&getContext());

    // The pattern matches the func operations and rewrites the func by
    // unrolling affine loop blocks sequentially and forwarding scalars after
    // each unroll.
    // Because the root operation of the pattern is not replaced, we limit the
    // number of rewrites and iterations to one.
    mlir::GreedyRewriteConfig config;
    config.maxIterations = 1;
    config.maxNumRewrites = 1;
    // FIXME: This was needed to target the first function. The maxIterations
    // and maxNumRewriters restriction means only one function can be iterated
    // on. If these values are >1, then only a single func is iterated on.
    config.useTopDownTraversal = true;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }

  mlir::StringRef getArgument() const final { return "unroll-and-forward"; }

 private:
  LogicalResult unrollAndForwardStores();
};

std::unique_ptr<Pass> createUnrollAndForwardStores() {
  return std::make_unique<UnrollAndForwardPass>();
}

}  // namespace heir
}  // namespace mlir