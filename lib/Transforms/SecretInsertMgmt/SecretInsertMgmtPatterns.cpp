#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"

#include <algorithm>
#include <cstdint>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

template <typename MulOp>
LogicalResult MultRelinearize<MulOp>::matchAndRewrite(
    MulOp mulOp, PatternRewriter& rewriter) const {
  Value result = mulOp.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return failure();
  }

  // if mul const, skip
  for (auto operand : mulOp.getOperands()) {
    auto secret = isSecret(operand, solver);
    if (!secret) {
      return failure();
    }
  }

  rewriter.setInsertionPointAfter(mulOp);
  auto relinearized =
      mgmt::RelinearizeOp::create(rewriter, mulOp.getLoc(), result);
  result.replaceAllUsesExcept(relinearized, {relinearized});

  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename MulOp>
LogicalResult ModReduceAfterMult<MulOp>::matchAndRewrite(
    MulOp mulOp, PatternRewriter& rewriter) const {
  Value result = mulOp.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return failure();
  }

  rewriter.setInsertionPointAfter(mulOp);
  auto modReduced = mgmt::ModReduceOp::create(rewriter, mulOp.getLoc(), result);
  result.replaceAllUsesExcept(modReduced, {modReduced});

  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult ModReduceBefore<Op>::matchAndRewrite(
    Op op, PatternRewriter& rewriter) const {
  // guard against secret::YieldOp
  if (op->getResults().size() > 0) {
    for (auto result : op->getResults()) {
      bool secret = isSecret(result, solver);
      if (!secret) {
        return failure();
      }
    }
  }
  // condition on result being secret

  auto maxLevel = 0;
  int64_t mulDepth = 0;
  SmallVector<OpOperand*, 2> secretOperands;
  getSecretOperands(op, secretOperands, solver);
  for (auto* operand : secretOperands) {
    auto levelState =
        solver->lookupState<LevelLattice>(operand->get())->getValue();
    if (!levelState.isInitialized()) {
      return failure();
    }
    auto mulDepthState =
        solver->lookupState<MulDepthLattice>(operand->get())->getValue();
    if (!mulDepthState.isInitialized()) {
      return failure();
    }

    auto level = levelState.getLevel();
    maxLevel = std::max(maxLevel, level);
    mulDepth = std::max(mulDepth, mulDepthState.getMulDepth());

    LLVM_DEBUG(llvm::dbgs() << "  ModReduceBefore: Operand: " << operand
                            << " Level: " << level << " MulDepth: "
                            << mulDepthState.getMulDepth() << "\n");
  }

  // first mulOp in the chain, skip
  if (!includeFirstMul && mulDepth == 0) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "ModReduceBefore: " << op
                          << " Level: " << maxLevel + 1 << "\n");

  SmallVector<Value, 2> secretOperandValues = llvm::to_vector(
      llvm::map_range(secretOperands, [](OpOperand* op) { return op->get(); }));
  // iterating over Values instead of OpOperands
  // because one Value can corresponds to multiple OpOperands
  for (auto operand : secretOperandValues) {
    rewriter.setInsertionPoint(op);
    auto managed = mgmt::ModReduceOp::create(rewriter, op.getLoc(), operand);
    op->replaceUsesOfWith(operand, managed);
  }

  // propagateIfChanged only push workitem to the worklist queue
  // actually execute the transfer for the new values
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult MatchCrossLevel<Op>::matchAndRewrite(
    Op op, PatternRewriter& rewriter) const {
  Value result = op.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return failure();
  }
  auto resultLevelState = solver->lookupState<LevelLattice>(result)->getValue();
  if (!resultLevelState.isInitialized()) {
    return failure();
  }
  auto resultLevel = resultLevelState.getLevel();

  bool inserted = false;
  SmallVector<OpOperand*, 2> secretOperands;
  getSecretOperands(op, secretOperands, solver);
  for (auto* operand : secretOperands) {
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
        managed = mgmt::LevelReduceOp::create(rewriter, op.getLoc(), managed,
                                              resultLevel - level - 1);
      }
      // make a different adjust scale each time
      // only after parameter selection can we decide the actual scale
      managed = mgmt::AdjustScaleOp::create(
          rewriter, op.getLoc(), managed,
          rewriter.getI64IntegerAttr((*idCounter)++));
      managed = mgmt::ModReduceOp::create(rewriter, op.getLoc(), managed);
      // NOTE that only at most one operand/Value will experience such
      // replacement. For op with two operands with same Value, such replace
      // won't happen.
      op->replaceUsesOfWith(operand->get(), managed);
    }
  }

  if (!inserted) {
    return failure();
  }
  // propagateIfChanged only push workitem to the worklist queue
  // actually execute the transfer for the new values
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult MatchCrossMulDepth<Op>::matchAndRewrite(
    Op op, PatternRewriter& rewriter) const {
  Value result = op.getResult();
  bool secret = isSecret(result, solver);
  if (!secret) {
    return failure();
  }

  SmallVector<OpOperand*, 2> secretOperands;
  getSecretOperands(op, secretOperands, solver);
  if (secretOperands.size() < 2) {
    return failure();
  }

  SmallVector<int64_t, 2> mulDepths;
  for (auto* operand : secretOperands) {
    auto mulDepthState =
        solver->lookupState<MulDepthLattice>(operand->get())->getValue();
    if (!mulDepthState.isInitialized()) {
      return failure();
    }
    auto mulDepth = mulDepthState.getMulDepth();
    mulDepths.push_back(mulDepth);
  }

  // only the input level can have mul depth mismatch.
  bool mismatch = (mulDepths[0] == 0 && mulDepths[1] == 1) ||
                  (mulDepths[0] == 1 && mulDepths[1] == 0);
  if (!mismatch) {
    return failure();
  }

  // for one operand being mulResult and another not,
  // we should match their scale by adding one adjust scale op
  for (auto i = 0; i < secretOperands.size(); i++) {
    auto* operand = secretOperands[i];
    auto mulDepth = mulDepths[i];
    if (!mulDepth) {
      rewriter.setInsertionPoint(op);
      Value managed = operand->get();
      // make a different adjust scale each time
      // only after parameter selection can we decide the actual scale
      managed = mgmt::AdjustScaleOp::create(
          rewriter, op.getLoc(), managed,
          rewriter.getI64IntegerAttr((*idCounter)++));
      op->replaceUsesOfWith(operand->get(), managed);
    }
  }

  // propagateIfChanged only push workitem to the worklist queue
  // actually execute the transfer for the new values
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult UseInitOpForPlaintextOperand<Op>::matchAndRewrite(
    Op op, PatternRewriter& rewriter) const {
  // If all results are non-secret and the operation is pure, nothing to do.
  // This handles common cases like index arithmetic within a loop.
  bool hasNoSecretResults = op->getResults().empty() ||
                            llvm::all_of(op->getResults(), [&](Value result) {
                              return !isSecret(result, solver);
                            });
  if (isMemoryEffectFree(op) && hasNoSecretResults) {
    return failure();
  }

  // insert mgmt::InitOp as an mgmt attribute placeholder for plaintext operand
  bool inserted = false;
  for (auto& operand : op->getOpOperands()) {
    bool secret = isSecret(operand.get(), solver);
    auto definingOp = operand.get().getDefiningOp();
    bool alreadyInitted =
        definingOp != nullptr && isa<mgmt::InitOp>(definingOp);
    if (!secret && !alreadyInitted) {
      rewriter.setInsertionPoint(op);
      auto initOp = mgmt::InitOp::create(
          rewriter, op.getLoc(), operand.get().getType(), operand.get());
      op->setOperand(operand.getOperandNumber(), initOp.getResult());
      inserted = true;
    }
  }
  if (!inserted) {
    return failure();
  }
  return success();
}

template <typename Op>
LogicalResult BootstrapWaterLine<Op>::matchAndRewrite(
    Op op, PatternRewriter& rewriter) const {
  auto levelLattice = solver->lookupState<LevelLattice>(op->getResult(0));
  if (!levelLattice->getValue().isInitialized()) {
    return failure();
  }

  auto level = levelLattice->getValue().getLevel();

  if (level < waterline) {
    return failure();
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
  auto bootstrap = mgmt::BootstrapOp::create(
      rewriter, op.getLoc(), op->getResultTypes(), op->getResult(0));
  op->getResult(0).replaceAllUsesExcept(bootstrap, {bootstrap});

  // greedy rewrite! note that we may get undeterministic insertion result
  // if we use different order of rewrites
  // currently walkAndApplyPatterns is deterministic
  solver->eraseAllStates();
  return solver->initializeAndRun(top);
}

// For all schemes
template struct BootstrapWaterLine<mgmt::ModReduceOp>;
template struct ModReduceBefore<secret::YieldOp>;
template struct UseInitOpForPlaintextOperand<tensor::ExtractSliceOp>;
template struct UseInitOpForPlaintextOperand<tensor::InsertSliceOp>;

// for BGV (integer ops)
template struct MatchCrossLevel<arith::AddIOp>;
template struct MatchCrossLevel<arith::MulIOp>;
template struct MatchCrossLevel<arith::SubIOp>;
template struct MatchCrossMulDepth<arith::AddIOp>;
template struct MatchCrossMulDepth<arith::MulIOp>;
template struct MatchCrossMulDepth<arith::SubIOp>;
template struct ModReduceAfterMult<arith::MulIOp>;
template struct ModReduceBefore<arith::MulIOp>;
template struct MultRelinearize<arith::MulIOp>;
template struct UseInitOpForPlaintextOperand<arith::AddIOp>;
template struct UseInitOpForPlaintextOperand<arith::MulIOp>;
template struct UseInitOpForPlaintextOperand<arith::SubIOp>;

// for CKKS (floating point ops)
template struct MatchCrossLevel<arith::AddFOp>;
template struct MatchCrossLevel<arith::MulFOp>;
template struct MatchCrossLevel<arith::SubFOp>;
template struct MatchCrossMulDepth<arith::AddFOp>;
template struct MatchCrossMulDepth<arith::MulFOp>;
template struct MatchCrossMulDepth<arith::SubFOp>;
template struct ModReduceAfterMult<arith::MulFOp>;
template struct ModReduceBefore<arith::MulFOp>;
template struct MultRelinearize<arith::MulFOp>;
template struct UseInitOpForPlaintextOperand<arith::AddFOp>;
template struct UseInitOpForPlaintextOperand<arith::MulFOp>;
template struct UseInitOpForPlaintextOperand<arith::SubFOp>;

}  // namespace heir
}  // namespace mlir
