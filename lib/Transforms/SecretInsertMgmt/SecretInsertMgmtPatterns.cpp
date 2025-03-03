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
  // use map in case we have same operands
  DenseMap<Value, LevelState::LevelType> operandsInsertLevel;
  for (auto operand : op.getOperands()) {
    bool secret = isSecret(operand, solver);
    if (!secret) {
      continue;
    }
    auto levelLattice = solver->lookupState<LevelLattice>(operand)->getValue();
    if (!levelLattice.isInitialized()) {
      return failure();
    }
    auto isMulResultLattice =
        solver->lookupState<MulResultLattice>(operand)->getValue();
    if (!isMulResultLattice.isInitialized()) {
      return failure();
    }

    auto level = levelLattice.getLevel();
    operandsInsertLevel[operand] = level;
    maxLevel = std::max(maxLevel, level);
    isMulResult |= isMulResultLattice.getIsMulResult();

    LLVM_DEBUG(llvm::dbgs() << "  ModReduceBefore: Operand: " << operand
                            << " Level: " << level << " isMulresult "
                            << isMulResultLattice.getIsMulResult() << "\n");
  }

  auto resultLevel = maxLevel;
  // for other op it is only mod reduce when operand mismatch level
  if (isMul) {
    // if includeFirstMul, we always mod reduce before
    // else, check if it is a mul result
    if (includeFirstMul || (!includeFirstMul && isMulResult)) {
      // Inserting mod reduce before mulOp means resultLevel = maxLevel +
      // 1
      resultLevel = maxLevel + 1;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "ModReduceBefore: " << op
                          << " Level: " << resultLevel << "\n");

  bool inserted = false;
  for (auto [operand, operandLevel] : operandsInsertLevel) {
    Value managed = operand;
    rewriter.setInsertionPoint(op);
    for (auto i = 0; i < resultLevel - operandLevel; ++i) {
      inserted = true;
      managed = rewriter.create<mgmt::ModReduceOp>(op.getLoc(), managed);
    };
    op->replaceUsesOfWith(operand, managed);
  }

  // when actually created new op, re-run the solver
  // otherwise performance issue
  if (inserted) {
    // propagateIfChanged only push workitem to the worklist queue
    // actually execute the transfer for the new values
    solver->eraseAllStates();
    return solver->initializeAndRun(top);
  }
  return success();
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

// isMul = true
template struct ModReduceBefore<arith::MulIOp>;
// extract = mulplain + rotate for annotated slot_extract = true
// TODO(#1174): decide packing earlier in the pipeline instead of annotation
template struct ModReduceBefore<tensor::ExtractOp>;
template struct ModReduceBefore<secret::YieldOp>;
// isMul = false
template struct ModReduceBefore<arith::AddIOp>;
template struct ModReduceBefore<arith::SubIOp>;

// for B/FV
template struct RemoveOp<mgmt::ModReduceOp>;

// for CKKS
template struct MultRelinearize<arith::MulFOp>;

// isMul = true
template struct ModReduceBefore<arith::MulFOp>;
// isMul = false
template struct ModReduceBefore<arith::AddFOp>;
template struct ModReduceBefore<arith::SubFOp>;

template struct BootstrapWaterLine<mgmt::ModReduceOp>;

}  // namespace heir
}  // namespace mlir
