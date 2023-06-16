#include <deque>
#include <queue>

#include "include/Conversion/MemrefToArith/MemrefToArith.h"
#include "include/Conversion/MemrefToArith/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h" // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h" // from @llvm-project
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
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h" // from @llvm-project
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

using ::mlir::Operation;
using ::mlir::Value;
using ::mlir::affine::AffineForOp;
using ::mlir::affine::AffineLoadOp;
using ::mlir::affine::AffineReadOpInterface;
using ::mlir::affine::AffineStoreOp;
using ::mlir::affine::AffineWriteOpInterface;
using ::mlir::affine::MemRefAccess;
using ::mlir::arith::ConstantOp;
using ::mlir::func::FuncOp;
using ::mlir::memref::AllocOp;
using ::mlir::memref::CollapseShapeOp;
using ::mlir::memref::ExpandShapeOp;
using ::mlir::memref::ExtractStridedMetadataOp;
using ::mlir::memref::GetGlobalOp;
using ::mlir::memref::ReinterpretCastOp;
using ::mlir::memref::SubViewOp;

// A map key combining the "raw" underlying memref (passed through any
// reinterpret_cast ops or similar renames) and the flattened access index
// (evaluating any affine maps), used in caches for the pass.
typedef std::pair<Value, uint64_t> NormalizedMemrefAccess;

class NormalizedMemrefAccessHash {
 public:
  size_t operator()(const NormalizedMemrefAccess &pair) const {
    return mlir::hash_value(pair.first) ^ std::hash<uint64_t>()(pair.second);
  }
};

// For each normalized input key, this contains the store ops
// that writes to the corresponding underlying memory location, even if the
// memref being written to has a different name/metadata.
typedef std::unordered_multimap<NormalizedMemrefAccess, Operation *,
                                NormalizedMemrefAccessHash>
    StoreMap;

// Scan all operations that could store a value to the given memref at any
// index, possibly transitively through renaming the memref via subview,
// collapse, expand, reinterpret_cast, or extract_strided_metadata. Add the
// resulting store ops to the `stores` deque.
static void collectAllTransitiveStoreOps(
    std::deque<AffineWriteOpInterface> &stores, Value &sourceMemRef) {
  std::queue<Operation *> users;
  for (auto user : sourceMemRef.getUsers()) {
    users.push(user);
  }

  for (; !users.empty(); users.pop()) {
    Operation *user = users.front();
    bool done =
        llvm::TypeSwitch<Operation &, bool>(*user)
            .Case<CollapseShapeOp, ExpandShapeOp, ReinterpretCastOp, SubViewOp>(
                [&](auto op) {
                  for (auto user : op.getResult().getUsers()) {
                    users.push(user);
                  }
                  return true;
                })
            .Case<ExtractStridedMetadataOp>([&](auto op) {
              for (auto user : op.getResults()[0].getUsers()) {
                users.push(user);
              }
              return true;
            })
            .Case<AffineStoreOp>([&](auto op) { return false; })
            .Default([&](Operation &) { return true; });

    if (done) {
      continue;
    }

    stores.push_back(llvm::cast<AffineWriteOpInterface>(user));
  }
}

// Extract a static access index from the MemRefAccess, and flatten it to a 1-d
// index of the underlying address space.
static FailureOr<int64_t> materializeAndFlatten(MemRefAccess access,
                                                Operation &op,
                                                MemRefType type) {
  auto optionalAccessIndices = materialize(access);
  if (!optionalAccessIndices) {
    op.emitWarning() << "Could not materialize access indices";
    return failure();
  }
  auto materialized = optionalAccessIndices.value();
  llvm::SmallVector<int64_t> castIndices;
  for (auto ndx : materialized) {
    castIndices.push_back((int64_t)ndx);
  }
  auto [strides, offset] = getStridesAndOffset(type);
  return flattenIndex(castIndices, strides, offset);
}

static FailureOr<int64_t> materializeAndFlattenAccessIndex(
    AffineWriteOpInterface op) {
  MemRefAccess storeAccess(op);
  return materializeAndFlatten(
      storeAccess, *op, llvm::cast<MemRefType>(op.getMemRef().getType()));
}

static FailureOr<int64_t> materializeAndFlattenAccessIndex(
    AffineReadOpInterface op) {
  MemRefAccess loadAccess(op);
  return materializeAndFlatten(
      loadAccess, *op, llvm::cast<MemRefType>(op.getMemRef().getType()));
}

// Search backwards through the IR to find the original memref that the input
// memref refers to. This can result in either a memref created by an alloc op
// or a function (block) argument memref.
static Value findSourceMemRef(Value memRef) {
  Value sourceMemRef = memRef;
  // The interesting cases:
  //
  // 1. If the memref is a block argument, then getDefiningOp returns null, and
  // we can exit because the memref we have is the source.
  // 2. The defining op is a memref.alloc, in which case we'd infinitely loop.
  // So we break out by swapping in nullptr for the "defining op."
  Operation *op = memRef.getDefiningOp();
  while (op != nullptr) {
    auto [value, newOp] =
        llvm::TypeSwitch<Operation &, std::pair<Value, Operation *>>(*op)
            .Case<ReinterpretCastOp, ExtractStridedMetadataOp, SubViewOp>(
                [&](auto op) {
                  return std::make_pair(op.getSource(),
                                        op.getSource().getDefiningOp());
                })
            .Case<AllocOp>([&](auto op) {
              return std::make_pair(op.getMemref(), nullptr);
            })
            .Case<ExpandShapeOp, CollapseShapeOp>([&](auto op) {
              return std::make_pair(op.getSrc(), op.getSrc().getDefiningOp());
            })
            .Default([&](Operation &) {
              llvm_unreachable("Unknown defining op");
              return std::make_pair(nullptr, nullptr);
            });

    op = newOp;
    sourceMemRef = value;
  }

  return sourceMemRef;
}

// Go through all ops between from and to and add all stores to the storeMap.
static LogicalResult updateStoreMap(Operation *from, Operation *to,
                                    StoreMap &storeMap) {
  for (Operation *op = from; op != to; op = op->getNextNode()) {
    auto storeOp = dyn_cast<AffineStoreOp>(op);
    if (!storeOp) {
      continue;
    }

    auto res = materializeAndFlattenAccessIndex(storeOp);
    if (failed(res)) {
      storeOp.emitWarning()
          << "Found storeOp with unmaterializable access index= " << storeOp;
      return failure();
    }
    int64_t storeAccessIndex = res.value();

    // FIXME: Consider building a small map to avoid recomputing this.
    Value storeSourceMemref = findSourceMemRef(storeOp.getMemRef());

    // The original memref for the store is the same as the original memref
    // for the load, because the store was found by searching for all users of
    // loadSourceMemref.
    NormalizedMemrefAccess storeIndexKey =
        std::make_pair(storeSourceMemref, storeAccessIndex);
    storeMap.insert(std::make_pair(storeIndexKey, storeOp));
  }
  return success();
}

// For a given load op that is not contained in any loop, and whose access
// indices are statically constant, find the last store op that stores to the
// corresponding memory location, and forward the stored value to the load. If
// the load is loading from a function argument memref, then collapse sequences
// of subview/expand/collapse/etc so that the target load is loading directly
// from the argument memref.
static LogicalResult forwardFullyUnrolledStoreToLoad(
    AffineReadOpInterface loadOp, std::vector<Operation *> &opsToErase,
    StoreMap &storeMap) {
  std::optional<Operation *> storeOpOrNull;
  auto loadMemRef = loadOp.getMemRef();
  Operation *loadDefiningOp = loadMemRef.getDefiningOp();

  if (loadDefiningOp && dyn_cast<GetGlobalOp>(loadDefiningOp)) {
    // A later pass handles forwarding from getglobal.
    return failure();
  }

  auto res = materializeAndFlattenAccessIndex(loadOp);
  if (failed(res)) {
    loadOp.emitWarning() << "Found loadOp with unmaterializable access index";
    return failure();
  }
  int64_t loadAccessIndex = res.value();

  Value loadSourceMemref = findSourceMemRef(loadMemRef);
  bool isBlockArgument = isa<BlockArgument>(loadSourceMemref);
  NormalizedMemrefAccess loadIndexKey =
      std::make_pair(loadSourceMemref, loadAccessIndex);

  if (isBlockArgument && loadSourceMemref == loadMemRef) {
    // This is a loadOp that loads directly from the function argument.
    return success();
  }

  // Check if a corresponding storeOp already exists in the storeMap.
  auto storeRes = storeMap.equal_range(loadIndexKey);
  for (auto it = storeRes.first; it != storeRes.second; ++it) {
    // Retrieve the latest store operation that's before the load operation.
    if (!storeOpOrNull.has_value() ||
        (storeOpOrNull.value()->isBeforeInBlock(it->second) &&
         (it->second)->isBeforeInBlock(loadOp))) {
      storeOpOrNull = it->second;
    }
  }

  if (!isBlockArgument && !storeOpOrNull.has_value()) {
    loadOp.emitWarning() << "Store op is null!"
                         << "; loadOp=" << loadOp
                         << "; loadAccessIndex=" << loadAccessIndex;
    llvm_unreachable("Should always be able to find a store op. File a bug.");
    return failure();
  }

  if (isBlockArgument) {
    // In this case, the load cannot be completely removed, but instead can be
    // replaced with a load from the original argument memref at the appropriate
    // index.
    const auto [endingStrides, endingOffset] =
        getStridesAndOffset(llvm::cast<MemRefType>(loadSourceMemref.getType()));

    ImplicitLocOpBuilder b(loadOp->getLoc(), loadOp);
    llvm::SmallVector<Value> indexValues;
    for (auto ndx :
         unflattenIndex(loadAccessIndex, endingStrides, endingOffset)) {
      Value ndxValue = b.create<ConstantOp>(b.getIndexAttr(ndx));
      indexValues.push_back(ndxValue);
    }

    auto newLoadOp = b.create<AffineLoadOp>(loadSourceMemref, indexValues);
    loadOp.getValue().replaceAllUsesWith(newLoadOp.getValue());
    opsToErase.push_back(loadOp);
    return success();
  } else {
    // In this case, the stored value can be forwarded directly to the load.
    Value storeVal =
        cast<AffineWriteOpInterface>(storeOpOrNull.value()).getValueToStore();
    // Check if 2 values have the same shape. This is needed for affine
    // vector loads and stores.
    if (storeVal.getType() != loadOp.getValue().getType()) {
      return failure();
    }
    // Replace the load value with the store value.
    loadOp.getValue().replaceAllUsesWith(storeVal);

    // We can't delete the StoreOp at this point (or any point), because we may
    // not fully remove all the places that load from it. Leave it to a future
    // pass to delete all stores to memory locations that have no corresponding
    // loads.

    opsToErase.push_back(loadOp);
    return success();
  }
}

}  // namespace

class UnrollAndForwardPattern final : public RewritePattern {
 public:
  explicit UnrollAndForwardPattern(MLIRContext *context)
      : RewritePattern(FuncOp::getOperationName(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    auto func = mlir::cast<FuncOp>(op);

    // Hold an intermediate computation of getFlattenedAccessIndex to avoid
    // repeated computations of MemRefAccess::getAccessMap
    std::vector<Operation *> opsToErase;

    // Hold a multi-map indexing all fully unrolled store operations by their
    // [Memref Value and flat index].
    StoreMap storeMap;
    // Add any stores to the store map that are not contained in any for loops.
    Operation &start = *func->getRegion(0).getOps().begin();
    auto end = *func.getOps<func::ReturnOp>().begin();
    if (failed(updateStoreMap(&start, end.getOperation(), storeMap))) {
      return failure();
    }

    rewriter.startRootUpdate(func);
    auto outerLoops = func.getOps<AffineForOp>();
    for (auto root : llvm::make_early_inc_range(outerLoops)) {
      // Update the positions of the operations before and after the outer for
      // loop.
      auto prevNode = root->getPrevNode();
      auto nextNode = root->getNextNode();

      SmallVector<AffineForOp> nestedLoops;
      mlir::affine::getPerfectlyNestedLoops(nestedLoops, root);
      nestedLoops[0].getBody(0)->walk<WalkOrder::PostOrder>(
          [&](AffineForOp forOp) {
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

      // Update the storeMap indexing all newly unrolled stores from the end of
      // the last loop to the end of the current loop.
      if (failed(updateStoreMap(prevNode, nextNode, storeMap)))
        return failure();

      //  Walk all load's and perform store to load forwarding.
      func.walk<WalkOrder::PreOrder>([&](AffineReadOpInterface loadOp) {
        if (loadOp->getParentOp() != nextNode->getParentOp() ||
            nextNode->isBeforeInBlock(loadOp)) {
          // Only iterate on the loads we just unravelled. Because we walk
          // in pre-order, we can interrupt the walk at this point.
          return WalkResult::interrupt();
        }

        if (loadOp->getParentOp() != prevNode->getParentOp() ||
            loadOp->isBeforeInBlock(prevNode)) {
          // Don't process any loads prev to the currently inspected block
          // that failed to forward, though ideally there should be none.
          return WalkResult::skip();
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

    // At this point, all the loops are unrolled, and all the load ops
    // that were within those loops, that could have been forwarded,
    // have been forwarded. However, there may still be load ops that
    // originated outside of any for loop that can still be forwarded.
    // So we need another pass over those load ops. However, this will also
    // re-process any load ops from get_globals, as well as loads from block
    // arguments. So those will report failure and be skipped.
    auto remainingLoads = func.getOps<AffineLoadOp>();
    for (auto loadOp : llvm::make_early_inc_range(remainingLoads)) {
      if (failed(
              forwardFullyUnrolledStoreToLoad(loadOp, opsToErase, storeMap))) {
        continue;
      }
    }
    for (auto *op : opsToErase) {
      op->erase();
    }
    opsToErase.clear();

    rewriter.finalizeRootUpdate(func);
    return success();
  };
};

// UnrollAndForwardPass intends to forward scalars.
struct UnrollAndForwardPass
    : public PassWrapper<UnrollAndForwardPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() {
    ConversionTarget target(getContext());

    RewritePatternSet patterns(&getContext());
    patterns.add<UnrollAndForwardPattern>(&getContext());

    // The pattern matches the func operations and rewrites the func by
    // unrolling affine loop blocks sequentially and forwarding scalars after
    // each unroll.
    // Because the root operation of the pattern is not replaced, we limit the
    // number of rewrites and iterations to one.
    GreedyRewriteConfig config;
    config.maxIterations = 1;
    config.maxNumRewrites = 1;
    // TODO(b/286582589): This was needed to target the first function. The
    // maxIterations and maxNumRewriters restriction means only one function can
    // be iterated on. If these values are >1, then only a single func is
    // iterated on.
    config.useTopDownTraversal = true;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }

  StringRef getArgument() const final { return "unroll-and-forward"; }

 private:
  LogicalResult unrollAndForwardStores();
};

std::unique_ptr<Pass> createUnrollAndForwardStoresPass() {
  return std::make_unique<UnrollAndForwardPass>();
}

}  // namespace heir
}  // namespace mlir
