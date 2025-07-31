#include "lib/Dialect/Lattigo/Transforms/AllocToInplace.h"

#include <algorithm>
#include <utility>

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/Tablegen/InplaceOpInterface.h"
#include "mlir/include/mlir/Analysis/Liveness.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

// There are two types of Value in the IR:
// 1. Storage: the actual memory allocated for the value
// 2. Referring Value: the value that refers to the storage (e.g., the
// returned SSA of an inplace operation)
//
// This class is similar to Disjoint-set data structure.
// Each Storage is the root, and all Referring Values are in its set.
//
// At the beginning StorageInfo should be initialized based on current relation
// in program.
//
// During rewriting, when we find available Storage for an AllocOp, we replace
// it with an InplaceOp and update the StorageInfo by merging the Storage of the
// AllocOp to the available Storage.
//
// This allows a mix of AllocOp and InplaceOp in input IR for the pass.
class StorageInfo {
 public:
  StorageInfo() = default;

  void addStorage(Value value) {
    if (mlir::isa<RLWECiphertextType>(value.getType())) {
      storageToReferringValues[value] = {};
    }
  }

  void addReferringValue(Value storage, Value value) {
    storageToReferringValues[storage].push_back(value);
  }

 private:
  // maintenance should be called internally
  void removeStorage(Value value) { storageToReferringValues.erase(value); }

  void mergeStorage(Value from, Value to) {
    storageToReferringValues[to].reserve(storageToReferringValues[to].size() +
                                         storageToReferringValues[from].size());
    storageToReferringValues[to].insert(storageToReferringValues[to].end(),
                                        storageToReferringValues[from].begin(),
                                        storageToReferringValues[from].end());
    removeStorage(from);
  }

 public:
  // User API
  Value getStorageFromValue(Value value) const {
    for (auto &[storage, values] : storageToReferringValues) {
      if (value == storage) {
        return storage;
      }
      for (auto referringValue : values) {
        if (value == referringValue) {
          return storage;
        }
      }
    }
    return Value();
  }

  // Greedily use the first storage.
  //
  // This greedy policy is optimal in terms of memory usage in that
  // 1. All dead values for this operation are dead for later operations so
  // they are equivalent, which means the first dead value is enough.
  // 2. If we decide not to use inplace for this operation, but allocate a new
  // value, in the hope that later operation can benefit from the reserved value
  // of this decision. Later operation actually can always allocate a new value
  // so the memory usage is not affected by this operation's local decision.
  //
  // However, this might not be optimal in terms of cache-friendliness for
  // various accelerators. One basic optimization is to use the dead value that
  // is closest to the current operation in the block. But as we do not have the
  // information of the memory layout, we do not implement this optimization.
  Value getAvailableStorage(Operation *op, Liveness *liveness) const {
    Value availableStorage;
    for (auto &[storage, values] : storageToReferringValues) {
      // storage and all referring values are dead
      if (std::all_of(
              values.begin(), values.end(),
              [&](Value value) { return liveness->isDeadAfter(value, op); }) &&
          liveness->isDeadAfter(storage, op)) {
        availableStorage = storage;
        break;
      }
    }
    return availableStorage;
  }

  void replaceAllocWithInplace(Operation *oldOp, Operation *newOp,
                               Value storage) {
    // add newly created referring value
    for (auto result : newOp->getResults()) {
      addReferringValue(storage, result);
    }
    // remove storage of old op
    for (auto result : oldOp->getResults()) {
      mergeStorage(result, storage);
    }
  }

 private:
  DenseMap<Value, SmallVector<Value>> storageToReferringValues;
};

template <typename BinOp, typename InplaceOp>
struct ConvertBinOp : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  ConvertBinOp(mlir::MLIRContext *context, Liveness *liveness,
               DenseMap<Block *, StorageInfo> *blockToStorageInfo)
      : OpRewritePattern<BinOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(BinOp op,
                                PatternRewriter &rewriter) const override {
    auto &storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return failure();
    }

    // InplaceOp has the form: output = InplaceOp(evaluator, lhs, rhs,
    // inplace) where inplace is the actual output but for SSA form we need to
    // return a new value
    auto inplaceOp = InplaceOp::create(
        rewriter, op.getLoc(), op.getOperand(1).getType(), op.getOperand(0),
        op.getOperand(1), op.getOperand(2), storage);

    // Update storage info, which must happen before the op is removed
    storageInfo.replaceAllocWithInplace(op, inplaceOp, storage);

    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness *liveness;
  DenseMap<Block *, StorageInfo> *blockToStorageInfo;
};

template <typename UnaryOp, typename InplaceOp>
struct ConvertUnaryOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  ConvertUnaryOp(mlir::MLIRContext *context, Liveness *liveness,
                 DenseMap<Block *, StorageInfo> *blockToStorageInfo)
      : OpRewritePattern<UnaryOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const override {
    auto &storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return failure();
    }

    // InplaceOp has the form: output = InplaceOp(evaluator, lhs, inplace)
    // where inplace is the actual output but for SSA form we need to return a
    // new value
    auto inplaceOp =
        InplaceOp::create(rewriter, op.getLoc(), op.getOperand(1).getType(),
                          op.getOperand(0), op.getOperand(1), storage);

    storageInfo.replaceAllocWithInplace(op, inplaceOp, storage);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness *liveness;
  DenseMap<Block *, StorageInfo> *blockToStorageInfo;
};

template <typename RotateOp, typename InplaceOp>
struct ConvertRotateOp : public OpRewritePattern<RotateOp> {
  using OpRewritePattern<RotateOp>::OpRewritePattern;

  ConvertRotateOp(mlir::MLIRContext *context, Liveness *liveness,
                  DenseMap<Block *, StorageInfo> *blockToStorageInfo)
      : OpRewritePattern<RotateOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(RotateOp op,
                                PatternRewriter &rewriter) const override {
    auto &storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return failure();
    }

    // InplaceOp has the form: output = InplaceOp(evaluator, lhs, inplace)
    // {offset} where inplace is the actual output but for SSA form we need to
    // return a new value
    auto inplaceOp = InplaceOp::create(
        rewriter, op.getLoc(), op.getOperand(1).getType(), op.getOperand(0),
        op.getOperand(1), storage, op.getOffset());

    // update storage info
    storageInfo.replaceAllocWithInplace(op, inplaceOp, storage);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness *liveness;
  DenseMap<Block *, StorageInfo> *blockToStorageInfo;
};

template <typename DropLevelOp, typename InplaceOp>
struct ConvertDropLevelOp : public OpRewritePattern<DropLevelOp> {
  using OpRewritePattern<DropLevelOp>::OpRewritePattern;

  ConvertDropLevelOp(mlir::MLIRContext *context, Liveness *liveness,
                     DenseMap<Block *, StorageInfo> *blockToStorageInfo)
      : OpRewritePattern<DropLevelOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(DropLevelOp op,
                                PatternRewriter &rewriter) const override {
    auto &storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return failure();
    }

    // InplaceOp has the form: output = InplaceOp(evaluator, lhs, inplace)
    // {levelToDrop} where inplace is the actual output but for SSA form we need
    // to return a new value
    auto inplaceOp = InplaceOp::create(
        rewriter, op.getLoc(), op.getOperand(1).getType(), op.getOperand(0),
        op.getOperand(1), storage, op.getLevelToDrop());

    // update storage info
    storageInfo.replaceAllocWithInplace(op, inplaceOp, storage);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness *liveness;
  DenseMap<Block *, StorageInfo> *blockToStorageInfo;
};

#define GEN_PASS_DEF_ALLOCTOINPLACE
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

struct AllocToInplace : impl::AllocToInplaceBase<AllocToInplace> {
  using AllocToInplaceBase::AllocToInplaceBase;

  void runOnOperation() override {
    Liveness liveness(getOperation());

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    DenseMap<Block *, StorageInfo> blockToStorageInfo;
    // Initialize each func block's storage
    getOperation()->walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }
      for (auto &block : funcOp.getBody().getBlocks()) {
        auto &storageInfo = blockToStorageInfo[&block];
        // arguments are storages
        for (auto arg : block.getArguments()) {
          storageInfo.addStorage(arg);
        }
        block.walk<WalkOrder::PreOrder>([&](Operation *op) {
          // inplace op will not allocate new memory, it produces referring
          // values
          if (auto inplaceOpInterface =
                  mlir::dyn_cast<InplaceOpInterface>(op)) {
            auto inplaceOperand =
                op->getOperand(inplaceOpInterface.getInplaceOperandIndex());
            auto storage = storageInfo.getStorageFromValue(inplaceOperand);
            if (storage) {
              for (auto result : op->getResults()) {
                storageInfo.addReferringValue(storage, result);
              }
            }
          } else {
            // alloc op results are storages
            for (auto result : op->getResults()) {
              storageInfo.addStorage(result);
            }
          }
        });
      }
    });

    patterns.add<
        // BGV
        ConvertBinOp<lattigo::BGVAddNewOp, lattigo::BGVAddOp>,
        ConvertBinOp<lattigo::BGVSubNewOp, lattigo::BGVSubOp>,
        ConvertBinOp<lattigo::BGVMulNewOp, lattigo::BGVMulOp>,
        ConvertUnaryOp<lattigo::BGVRelinearizeNewOp, lattigo::BGVRelinearizeOp>,
        ConvertUnaryOp<lattigo::BGVRescaleNewOp, lattigo::BGVRescaleOp>,
        ConvertRotateOp<lattigo::BGVRotateColumnsNewOp,
                        lattigo::BGVRotateColumnsOp>,
        // CKKS
        ConvertBinOp<lattigo::CKKSAddNewOp, lattigo::CKKSAddOp>,
        ConvertBinOp<lattigo::CKKSSubNewOp, lattigo::CKKSSubOp>,
        ConvertBinOp<lattigo::CKKSMulNewOp, lattigo::CKKSMulOp>,
        ConvertUnaryOp<lattigo::CKKSRelinearizeNewOp,
                       lattigo::CKKSRelinearizeOp>,
        ConvertUnaryOp<lattigo::CKKSRescaleNewOp, lattigo::CKKSRescaleOp>,
        ConvertRotateOp<lattigo::CKKSRotateNewOp, lattigo::CKKSRotateOp>,
        // RLWE
        ConvertUnaryOp<lattigo::RLWENegateNewOp, lattigo::RLWENegateOp>,
        ConvertDropLevelOp<lattigo::RLWEDropLevelNewOp,
                           lattigo::RLWEDropLevelOp>>(context, &liveness,
                                                      &blockToStorageInfo);

    // The greedy policy relies on the order of processing the operations.
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
