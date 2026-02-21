#include "lib/Dialect/Lattigo/Transforms/AllocToInPlace.h"

#include <utility>

#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/AllocToInPlaceUtils.h"
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

template <typename BinOp, typename InPlaceOp>
struct ConvertBinOp : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  ConvertBinOp(mlir::MLIRContext* context, Liveness* liveness,
               DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo)
      : OpRewritePattern<BinOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(BinOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, lhs, rhs,
    // inplace) where inplace is the actual output but for SSA form we need to
    // return a new value
    auto inplaceOp = InPlaceOp::create(
        rewriter, op.getLoc(), op.getOperand(1).getType(), op.getOperand(0),
        op.getOperand(1), op.getOperand(2), storage);

    // Update storage info, which must happen before the op is removed
    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);

    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
};

template <typename UnaryOp, typename InPlaceOp>
struct ConvertUnaryOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  ConvertUnaryOp(
      mlir::MLIRContext* context, Liveness* liveness,
      DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo)
      : OpRewritePattern<UnaryOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, lhs, inplace)
    // where inplace is the actual output but for SSA form we need to return a
    // new value
    auto inplaceOp =
        InPlaceOp::create(rewriter, op.getLoc(), op.getOperand(1).getType(),
                          op.getOperand(0), op.getOperand(1), storage);

    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
};

template <typename RotateOp, typename InPlaceOp>
struct ConvertRotateOp : public OpRewritePattern<RotateOp> {
  using OpRewritePattern<RotateOp>::OpRewritePattern;

  ConvertRotateOp(
      mlir::MLIRContext* context, Liveness* liveness,
      DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo)
      : OpRewritePattern<RotateOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(RotateOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, input, inplace,
    // shift?, offset?) where inplace is the actual output but for SSA form we
    // need to return a new value Handle both shift SSA value and offset
    // attribute
    Value shift = op.getShift();
    auto inplaceOp = InPlaceOp::create(
        rewriter, op.getLoc(), op.getInput().getType(), op.getEvaluator(),
        op.getInput(), storage, shift, /*offset=*/nullptr);

    // update storage info
    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
};

template <typename DropLevelOp, typename InPlaceOp>
struct ConvertDropLevelOp : public OpRewritePattern<DropLevelOp> {
  using OpRewritePattern<DropLevelOp>::OpRewritePattern;

  ConvertDropLevelOp(
      mlir::MLIRContext* context, Liveness* liveness,
      DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo)
      : OpRewritePattern<DropLevelOp>(context),
        liveness(liveness),
        blockToStorageInfo(blockToStorageInfo) {}

  LogicalResult matchAndRewrite(DropLevelOp op,
                                PatternRewriter& rewriter) const override {
    auto& storageInfo = (*blockToStorageInfo)[op->getBlock()];
    auto storage = storageInfo.getAvailableStorage(op, liveness);
    if (!storage) {
      return rewriter.notifyMatchFailure(op, "no available storage found");
    }

    // InPlaceOp has the form: output = InPlaceOp(evaluator, lhs, inplace)
    // {levelToDrop} where inplace is the actual output but for SSA form we need
    // to return a new value
    auto inplaceOp = InPlaceOp::create(
        rewriter, op.getLoc(), op.getOperand(1).getType(), op.getOperand(0),
        op.getOperand(1), storage, op.getLevelToDrop());

    // update storage info
    storageInfo.replaceAllocWithInPlace(op, inplaceOp, storage);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  DenseMap<Block*, CallerProvidedStorageInfo>* blockToStorageInfo;
};

#define GEN_PASS_DEF_ALLOCTOINPLACE
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

struct AllocToInPlace : impl::AllocToInPlaceBase<AllocToInPlace> {
  using AllocToInPlaceBase::AllocToInPlaceBase;

  void runOnOperation() override {
    Liveness liveness(getOperation());

    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    DenseMap<Block*, CallerProvidedStorageInfo> blockToStorageInfo =
        initializeAllocToInPlaceBlockStorage<RLWECiphertextType>(
            getOperation());

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
