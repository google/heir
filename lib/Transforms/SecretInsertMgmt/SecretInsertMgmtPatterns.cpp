#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"

#include <algorithm>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulResultAnalysis/MulResultAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

template <typename MulOp>
LogicalResult MultRelinearize<MulOp>::matchAndRewrite(
    MulOp mulOp, PatternRewriter &rewriter) const {
  Value result = mulOp.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return success();
  }

  // if mul const, skip
  for (auto operand : mulOp.getOperands()) {
    auto secret = isSecret(operand, solver);
    if (!secret) {
      return success();
    }
  }

  rewriter.setInsertionPointAfter(mulOp);
  auto relinearized =
      rewriter.create<mgmt::RelinearizeOp>(mulOp.getLoc(), result);
  result.replaceAllUsesExcept(relinearized, {relinearized});

  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename MulOp>
LogicalResult ModReduceAfterMult<MulOp>::matchAndRewrite(
    MulOp mulOp, PatternRewriter &rewriter) const {
  Value result = mulOp.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return success();
  }

  // guard against tensor::ExtractOp with slot_extract = false
  // TODO(#1174): decide packing earlier in the pipeline instead of annotation
  // workaround
  if (isa<tensor::ExtractOp>(mulOp)) {
    auto slotExtractAttr = mulOp->getAttr("slot_extract");
    if (isa_and_nonnull<BoolAttr>(slotExtractAttr)) {
      // must be false!
      return success();
    }
  }

  rewriter.setInsertionPointAfter(mulOp);
  auto modReduced = rewriter.create<mgmt::ModReduceOp>(mulOp.getLoc(), result);
  result.replaceAllUsesExcept(modReduced, {modReduced});

  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult ModReduceBefore<Op>::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  // guard against secret::YieldOp
  if (op->getResults().size() > 0) {
    for (auto result : op->getResults()) {
      bool secret = isSecret(result, solver);
      if (!secret) {
        return success();
      }
    }
  }
  // condition on result being secret

  // guard against tensor::ExtractOp with slot_extract = false
  // TODO(#1174): decide packing earlier in the pipeline instead of annotation
  // workaround
  if (isa<tensor::ExtractOp>(op)) {
    auto slotExtractAttr = op->getAttr("slot_extract");
    if (isa_and_nonnull<BoolAttr>(slotExtractAttr)) {
      // must be false!
      return success();
    }
  }

  auto maxLevel = 0;
  auto isMulResult = false;
  SmallVector<OpOperand *, 2> secretOperands;
  getSecretOperands(op, secretOperands, solver);
  for (auto *operand : secretOperands) {
    auto levelState =
        solver->lookupState<LevelLattice>(operand->get())->getValue();
    if (!levelState.isInitialized()) {
      return failure();
    }
    auto isMulResultState =
        solver->lookupState<MulResultLattice>(operand->get())->getValue();
    if (!isMulResultState.isInitialized()) {
      return failure();
    }

    auto level = levelState.getLevel();
    maxLevel = std::max(maxLevel, level);
    isMulResult |= isMulResultState.getIsMulResult();

    LLVM_DEBUG(llvm::dbgs() << "  ModReduceBefore: Operand: " << operand->get()
                            << " Level: " << level << " isMulresult "
                            << isMulResultState.getIsMulResult() << "\n");
  }

  // first mulOp in the chain, skip
  if (!includeFirstMul && !isMulResult) {
    return success();
  }

  auto resultLevel = maxLevel + 1;

  LLVM_DEBUG(llvm::dbgs() << "ModReduceBefore: " << op
                          << " Level: " << resultLevel << "\n");

  SmallVector<Value, 2> secretOperandValues = llvm::to_vector(
      llvm::map_range(secretOperands, [](OpOperand *op) { return op->get(); }));
  // iterating over Values instead of OpOperands
  // because one Value can corresponds to multiple OpOperands
  for (auto operand : secretOperandValues) {
    rewriter.setInsertionPoint(op);
    auto managed = rewriter.create<mgmt::ModReduceOp>(op.getLoc(), operand);
    op->replaceUsesOfWith(operand, managed);
  }

  // propagateIfChanged only push workitem to the worklist queue
  // actually execute the transfer for the new values
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult MatchCrossLevel<Op>::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  Value result = op.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return success();
  }
  auto resultLevelState = solver->lookupState<LevelLattice>(result)->getValue();
  if (!resultLevelState.isInitialized()) {
    return failure();
  }
  auto resultLevel = resultLevelState.getLevel();

  bool inserted = false;
  SmallVector<OpOperand *, 2> secretOperands;
  getSecretOperands(op, secretOperands, solver);
  for (auto *operand : secretOperands) {
    auto levelState =
        solver->lookupState<LevelLattice>(operand->get())->getValue();
    if (!levelState.isInitialized()) {
      return failure();
    }

    auto level = levelState.getLevel();
    if (level < resultLevel) {
      inserted = true;
      rewriter.setInsertionPoint(op);
      Value managed = operand->get();
      if (resultLevel - level > 1) {
        managed = rewriter.create<mgmt::LevelReduceOp>(op.getLoc(), managed,
                                                       resultLevel - level - 1);
      }
      // make a different adjust scale each time
      // only after parameter selection can we decide the actual scale
      managed = rewriter.create<mgmt::AdjustScaleOp>(
          op.getLoc(), managed, rewriter.getI64IntegerAttr((*scaleCounter)--),
          rewriter.getF64FloatAttr(0.0));
      managed = rewriter.create<mgmt::ModReduceOp>(op.getLoc(), managed);
      // NOTE that only at most one operand/Value will experience such
      // replacement. For op with two operands with same Value, such replace
      // won't happen.
      op->replaceUsesOfWith(operand->get(), managed);
    }
  }

  if (!inserted) {
    return success();
  }
  // propagateIfChanged only push workitem to the worklist queue
  // actually execute the transfer for the new values
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult MatchCrossMulDepth<Op>::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  Value result = op.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return success();
  }

  auto maxMulDepth = 0;
  SmallVector<OpOperand *, 2> secretOperands;
  getSecretOperands(op, secretOperands, solver);
  for (auto *operand : secretOperands) {
    auto mulDepthState =
        solver->lookupState<MulDepthLattice>(operand->get())->getValue();
    if (!mulDepthState.isInitialized()) {
      return failure();
    }
    auto mulDepth = mulDepthState.getMulDepth();
    maxMulDepth = std::max(maxMulDepth, mulDepth);
  }

  bool inserted = false;
  for (auto *operand : secretOperands) {
    auto mulDepthState =
        solver->lookupState<MulDepthLattice>(operand->get())->getValue();
    if (!mulDepthState.isInitialized()) {
      return failure();
    }

    auto mulDepth = mulDepthState.getMulDepth();
    if (mulDepth < maxMulDepth) {
      assert(maxMulDepth - mulDepth <= 1 &&
             "Level/MulDepth mismatch can be at most 1");
      inserted = true;
      rewriter.setInsertionPoint(op);
      Value managed = operand->get();
      // make a different adjust scale each time
      // only after parameter selection can we decide the actual scale
      managed = rewriter.create<mgmt::AdjustScaleOp>(
          op.getLoc(), managed, rewriter.getI64IntegerAttr((*scaleCounter)--),
          rewriter.getF64FloatAttr(0.0));
      op->replaceUsesOfWith(operand->get(), managed);
    }
  }

  if (!inserted) {
    return success();
  }
  // propagateIfChanged only push workitem to the worklist queue
  // actually execute the transfer for the new values
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult RemoveOp<Op>::matchAndRewrite(Op op,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceAllUsesWith(op->getResult(0), op->getOperand(0));
  rewriter.eraseOp(op);
  return success();
}

template <typename Op>
LogicalResult BootstrapWaterLine<Op>::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  auto levelLattice = solver->lookupState<LevelLattice>(op->getResult(0));
  if (!levelLattice->getValue().isInitialized()) {
    return failure();
  }

  auto level = levelLattice->getValue().getLevel();

  if (level < waterline) {
    return success();
  }
  if (level > waterline) {
    // should never met!
    LLVM_DEBUG(llvm::dbgs()
               << "BootstrapWaterLine: met " << op << " with level: " << level
               << " but waterline: " << waterline << "\n");
    return failure();
  }

  // insert mgmt::BootstrapOp after
  rewriter.setInsertionPointAfter(op);
  auto bootstrap = rewriter.create<mgmt::BootstrapOp>(
      op.getLoc(), op->getResultTypes(), op->getResult(0));
  op->getResult(0).replaceAllUsesExcept(bootstrap, {bootstrap});

  // greedy rewrite! note that we may get undeterministic insertion result
  // if we use different order of rewrites
  // currently walkAndApplyPatterns is deterministic
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

// for BGV
template struct MultRelinearize<arith::MulIOp>;

template struct ModReduceAfterMult<arith::MulIOp>;
// extract = mulplain + rotate for annotated slot_extract = true
// TODO(#1174): decide packing earlier in the pipeline instead of annotation
template struct ModReduceAfterMult<tensor::ExtractOp>;

template struct ModReduceBefore<arith::MulIOp>;
// extract = mulplain + rotate for annotated slot_extract = true
// TODO(#1174): decide packing earlier in the pipeline instead of annotation
template struct ModReduceBefore<tensor::ExtractOp>;
template struct ModReduceBefore<secret::YieldOp>;

template struct MatchCrossLevel<arith::MulIOp>;
template struct MatchCrossLevel<arith::AddIOp>;
template struct MatchCrossLevel<arith::SubIOp>;

template struct MatchCrossMulDepth<arith::MulIOp>;
template struct MatchCrossMulDepth<arith::AddIOp>;
template struct MatchCrossMulDepth<arith::SubIOp>;

// for B/FV
template struct RemoveOp<mgmt::ModReduceOp>;

// for CKKS
template struct MultRelinearize<arith::MulFOp>;

//// isMul = true
template struct ModReduceBefore<arith::MulFOp>;
//// isMul = false
// template struct ModReduceBefore<arith::AddFOp>;
// template struct ModReduceBefore<arith::SubFOp>;

template struct BootstrapWaterLine<mgmt::ModReduceOp>;

}  // namespace heir
}  // namespace mlir
