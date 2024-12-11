#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulResultAnalysis/MulResultAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

template <typename MulOp>
LogicalResult MultRelinearize<MulOp>::matchAndRewrite(
    MulOp mulOp, PatternRewriter &rewriter) const {
  Value result = mulOp.getResult();
  auto secret = solver->lookupState<SecretnessLattice>(result)->getValue();
  // if not secret, skip
  if (!secret.isInitialized() || !secret.getSecretness()) {
    return success();
  }

  // if mul const, skip
  for (auto operand : mulOp.getOperands()) {
    auto secretness =
        solver->lookupState<SecretnessLattice>(operand)->getValue();
    if (!secretness.isInitialized() || !secretness.getSecretness()) {
      return success();
    }
  }

  rewriter.setInsertionPointAfter(mulOp);
  auto relinearized =
      rewriter.create<mgmt::RelinearizeOp>(mulOp.getLoc(), result);
  result.replaceAllUsesExcept(relinearized, {relinearized});

  return solver->initializeAndRun(top);
}

template <typename Op>
LogicalResult ModReduceBefore<Op>::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  // guard against secret::YieldOp
  if (op->getResults().size() > 0) {
    for (auto result : op->getResults()) {
      auto secret = solver->lookupState<SecretnessLattice>(result)->getValue();
      if (!secret.isInitialized() || !secret.getSecretness()) {
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
    auto secretness =
        solver->lookupState<SecretnessLattice>(operand)->getValue();
    if (!secretness.isInitialized()) {
      return failure();
    }
    if (!secretness.getSecretness()) {
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
    return solver->initializeAndRun(top);
  }
  return success();
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

// for CKKS
template struct MultRelinearize<arith::MulFOp>;

// isMul = true
template struct ModReduceBefore<arith::MulFOp>;
// isMul = false
template struct ModReduceBefore<arith::AddFOp>;
template struct ModReduceBefore<arith::SubFOp>;

void annotateMgmtAttr(Operation *top) {
  auto mergeIntoMgmtAttr = [&](Attribute levelAttr, Attribute dimensionAttr) {
    auto level = cast<IntegerAttr>(levelAttr).getInt();
    auto dimension = cast<IntegerAttr>(dimensionAttr).getInt();
    auto mgmtAttr = mgmt::MgmtAttr::get(top->getContext(), level, dimension);
    return mgmtAttr;
  };
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    bool bodyContainsSecretGeneric = false;
    funcOp->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
      bodyContainsSecretGeneric = true;
      for (auto i = 0; i != genericOp.getBody()->getNumArguments(); ++i) {
        auto levelAttr = genericOp.removeArgAttr(i, "level");
        auto dimensionAttr = genericOp.removeArgAttr(i, "dimension");
        auto mgmtAttr = mergeIntoMgmtAttr(levelAttr, dimensionAttr);
        genericOp.setArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName, mgmtAttr);
      }

      genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->getNumResults() == 0) {
          return;
        }
        auto levelAttr = op->removeAttr("level");
        auto dimensionAttr = op->removeAttr("dimension");
        if (!levelAttr || !dimensionAttr) {
          return;
        }
        op->setAttr(mgmt::MgmtDialect::kArgMgmtAttrName,
                    mergeIntoMgmtAttr(levelAttr, dimensionAttr));
      });
    });

    // handle generic-less function body
    // default to level 0 and dimension 2
    // otherwise secret-to-<scheme> won't find the mgmt attr
    if (!bodyContainsSecretGeneric) {
      for (auto i = 0; i != funcOp.getNumArguments(); ++i) {
        auto argument = funcOp.getArgument(i);
        if (isa<secret::SecretType>(argument.getType())) {
          funcOp.setArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName,
                            mgmt::MgmtAttr::get(top->getContext(), 0, 2));
        }
      }
    }
  });
}

}  // namespace heir
}  // namespace mlir
