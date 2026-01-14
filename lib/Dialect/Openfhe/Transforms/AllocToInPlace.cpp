#include "lib/Dialect/Openfhe/Transforms/AllocToInPlace.h"

#include <cassert>
#include <utility>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/AllocToInPlaceUtils.h"
#include "mlir/include/mlir/Analysis/Liveness.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

// TODO(#2531): This only works on the LHS operand, but if the op is
// commutative, and *both* operands are ciphertexts, the RHS ciphertext may be
// available to be used in-place when the LHS is not usable for in-place.
template <typename BinOp, typename InPlaceOp>
struct ConvertBinOp : public OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;

  ConvertBinOp(mlir::MLIRContext* context, Liveness* liveness,
               OperandMutatedStorageInfo* storageInfo)
      : OpRewritePattern<BinOp>(context),
        liveness(liveness),
        storageInfo(storageInfo) {}

  LogicalResult matchAndRewrite(BinOp op,
                                PatternRewriter& rewriter) const override {
    Value mutatedValue = op.getLhs();
    if (isa<PlaintextType>(mutatedValue.getType())) {
      assert(isa<CiphertextType>(op.getRhs().getType()));
      mutatedValue = op.getRhs();
    }
    if (!storageInfo->isSafeToMutateInPlace(op, mutatedValue, liveness)) {
      return rewriter.notifyMatchFailure(
          op,
          "Not replacing op with in-place version because the mutated "
          "operand is still live");
    }

    auto inplaceOp =
        InPlaceOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                          op.getCryptoContext(), op.getLhs(), op.getRhs());
    storageInfo->replaceAllocWithInPlace(op, inplaceOp);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  OperandMutatedStorageInfo* storageInfo;
};

template <typename UnaryOp, typename InPlaceOp>
struct ConvertUnaryOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  ConvertUnaryOp(mlir::MLIRContext* context, Liveness* liveness,
                 OperandMutatedStorageInfo* storageInfo)
      : OpRewritePattern<UnaryOp>(context),
        liveness(liveness),
        storageInfo(storageInfo) {}

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter& rewriter) const override {
    if (!storageInfo->isSafeToMutateInPlace(op, op.getCiphertext(), liveness)) {
      return rewriter.notifyMatchFailure(
          op,
          "Not replacing op with in-place version because the mutated "
          "operand is still live");
    }

    auto inplaceOp =
        InPlaceOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                          op.getCryptoContext(), op.getCiphertext());

    storageInfo->replaceAllocWithInPlace(op, inplaceOp);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  OperandMutatedStorageInfo* storageInfo;
};

struct ConvertLevelReduceOp : public OpRewritePattern<LevelReduceOp> {
  using OpRewritePattern<LevelReduceOp>::OpRewritePattern;

  ConvertLevelReduceOp(mlir::MLIRContext* context, Liveness* liveness,
                       OperandMutatedStorageInfo* storageInfo)
      : OpRewritePattern<LevelReduceOp>(context),
        liveness(liveness),
        storageInfo(storageInfo) {}

  LogicalResult matchAndRewrite(LevelReduceOp op,
                                PatternRewriter& rewriter) const override {
    if (!storageInfo->isSafeToMutateInPlace(op, op.getCiphertext(), liveness)) {
      return rewriter.notifyMatchFailure(
          op,
          "Not replacing op with in-place version because the mutated "
          "operand is still live");
    }

    auto inplaceOp = LevelReduceInPlaceOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), op.getCryptoContext(),
        op.getCiphertext(), op.getLevelToDrop());
    storageInfo->replaceAllocWithInPlace(op, inplaceOp);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  OperandMutatedStorageInfo* storageInfo;
};

struct ConvertMulConstOp : public OpRewritePattern<MulConstOp> {
  using OpRewritePattern<MulConstOp>::OpRewritePattern;

  ConvertMulConstOp(mlir::MLIRContext* context, Liveness* liveness,
                    OperandMutatedStorageInfo* storageInfo)
      : OpRewritePattern<MulConstOp>(context),
        liveness(liveness),
        storageInfo(storageInfo) {}

  LogicalResult matchAndRewrite(MulConstOp op,
                                PatternRewriter& rewriter) const override {
    if (!storageInfo->isSafeToMutateInPlace(op, op.getCiphertext(), liveness)) {
      return rewriter.notifyMatchFailure(
          op,
          "Not replacing op with in-place version because the mutated "
          "operand is still live");
    }

    auto inplaceOp = MulConstInPlaceOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), op.getCryptoContext(),
        op.getCiphertext(), op.getConstant());
    storageInfo->replaceAllocWithInPlace(op, inplaceOp);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  OperandMutatedStorageInfo* storageInfo;
};

struct ConvertKeySwitchOp : public OpRewritePattern<KeySwitchOp> {
  using OpRewritePattern<KeySwitchOp>::OpRewritePattern;

  ConvertKeySwitchOp(mlir::MLIRContext* context, Liveness* liveness,
                     OperandMutatedStorageInfo* storageInfo)
      : OpRewritePattern<KeySwitchOp>(context),
        liveness(liveness),
        storageInfo(storageInfo) {}

  LogicalResult matchAndRewrite(KeySwitchOp op,
                                PatternRewriter& rewriter) const override {
    if (!storageInfo->isSafeToMutateInPlace(op, op.getCiphertext(), liveness)) {
      return rewriter.notifyMatchFailure(
          op,
          "Not replacing op with in-place version because the mutated "
          "operand is still live");
    }

    auto inplaceOp = KeySwitchInPlaceOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), op.getCryptoContext(),
        op.getCiphertext(), op.getEvalKey());
    storageInfo->replaceAllocWithInPlace(op, inplaceOp);
    rewriter.replaceOp(op, inplaceOp);
    return success();
  }

 private:
  Liveness* liveness;
  OperandMutatedStorageInfo* storageInfo;
};

#define GEN_PASS_DEF_ALLOCTOINPLACE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

struct AllocToInPlace : impl::AllocToInPlaceBase<AllocToInPlace> {
  using AllocToInPlaceBase::AllocToInPlaceBase;

  void runOnOperation() override {
    Liveness liveness(getOperation());
    OperandMutatedStorageInfo storageInfo;

    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ConvertBinOp<AddOp, AddInPlaceOp>,
                 ConvertBinOp<SubOp, SubInPlaceOp>,
                 ConvertBinOp<AddPlainOp, AddPlainInPlaceOp>,
                 ConvertBinOp<SubPlainOp, SubPlainInPlaceOp>,
                 ConvertUnaryOp<NegateOp, NegateInPlaceOp>,
                 ConvertUnaryOp<SquareOp, SquareInPlaceOp>,
                 ConvertUnaryOp<RelinOp, RelinInPlaceOp>,
                 ConvertUnaryOp<ModReduceOp, ModReduceInPlaceOp>,
                 ConvertMulConstOp, ConvertLevelReduceOp, ConvertKeySwitchOp>(
        context, &liveness, &storageInfo);

    // The greedy policy relies on the order of processing the operations.
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
