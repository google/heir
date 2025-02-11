#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult LevelAnalysis::visitOperation(
    Operation *op, ArrayRef<const LevelLattice *> operands,
    ArrayRef<LevelLattice *> results) {
  auto propagate = [&](Value value, const LevelState &state) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, LevelState(0));
        }
      })
      .Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        // implicitly ensure that the operand is secret
        const auto *operandLattice = operands[0];
        if (!operandLattice->getValue().isInitialized()) {
          return;
        }
        auto level = operandLattice->getValue().getLevel();
        propagate(modReduceOp.getResult(), LevelState(level + 1));
      })
      .Case<mgmt::BootstrapOp>([&](auto bootstrapOp) {
        // implicitly ensure that the result is secret
        // reset level to 0
        // TODO(#1207): reset level to currentLevel - bootstrapDepth
        propagate(bootstrapOp.getResult(), LevelState(0));
      })
      .Default([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        auto levelResult = 0;
        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(&op, secretOperands);
        for (auto *operand : secretOperands) {
          auto &levelState = getLatticeElement(operand->get())->getValue();
          if (!levelState.isInitialized()) {
            return;
          }
          levelResult = std::max(levelResult, levelState.getLevel());
        }

        for (auto result : secretResults) {
          propagate(result, LevelState(levelResult));
        }
      });
  return success();
}

void LevelAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const LevelLattice *> argumentLattices,
    ArrayRef<LevelLattice *> resultLattices) {
  auto callback = std::bind(&LevelAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<LevelState, LevelLattice>(
      call, argumentLattices, resultLattices, callback);
}

static int getMaxLevel(Operation *top, DataFlowSolver *solver) {
  auto maxLevel = 0;
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      if (!isSecret(op->getResult(0), solver)) {
        return;
      }
      // ensure result is secret
      auto level = solver->lookupState<LevelLattice>(op->getResult(0))
                       ->getValue()
                       .getLevel();
      maxLevel = std::max(maxLevel, level);
    });
  });
  return maxLevel;
}

void annotateLevel(Operation *top, DataFlowSolver *solver) {
  auto maxLevel = getMaxLevel(top, solver);

  auto getIntegerAttr = [&](int level) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), level);
  };

  // use L to 0 instead of 0 to L
  auto getLevel = [&](Value value) {
    return maxLevel -
           solver->lookupState<LevelLattice>(value)->getValue().getLevel();
  };

  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    for (auto i = 0; i != genericOp.getBody()->getNumArguments(); ++i) {
      auto blockArg = genericOp.getBody()->getArgument(i);
      auto level = getLevel(blockArg);
      genericOp.setArgAttr(i, "level", getIntegerAttr(level));
    }

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      if (!isSecret(op->getResult(0), solver)) {
        return;
      }
      auto level = getLevel(op->getResult(0));
      op->setAttr("level", getIntegerAttr(level));
    });
  });
}

LevelState::LevelType getLevelFromMgmtAttr(Value value) {
  Attribute attr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto genericOp = dyn_cast<secret::GenericOp>(parentOp);
    if (genericOp) {
      attr = genericOp.getArgAttr(blockArg.getArgNumber(),
                                  mgmt::MgmtDialect::kArgMgmtAttrName);
    }
  } else {
    auto *parentOp = value.getDefiningOp();
    attr = parentOp->getAttr(mgmt::MgmtDialect::kArgMgmtAttrName);
  }
  if (!mlir::isa<mgmt::MgmtAttr>(attr)) {
    assert(false && "MgmtAttr not found");
  }
  auto mgmtAttr = mlir::cast<mgmt::MgmtAttr>(attr);
  return mgmtAttr.getLevel();
}

}  // namespace heir
}  // namespace mlir
