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
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "LevelAnalysis"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// LevelAnalysis (Forward)
//===----------------------------------------------------------------------===//

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
      .Case<mgmt::LevelReduceOp>([&](auto levelReduceOp) {
        // implicitly ensure that the operand is secret
        const auto *operandLattice = operands[0];
        if (!operandLattice->getValue().isInitialized()) {
          return;
        }
        auto level = operandLattice->getValue().getLevel();
        propagate(levelReduceOp.getResult(),
                  LevelState(level + levelReduceOp.getLevelToDrop()));
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

//===----------------------------------------------------------------------===//
// LevelAnalysis (Backward)
//===----------------------------------------------------------------------===//

LogicalResult LevelAnalysisBackward::visitOperation(
    Operation *op, ArrayRef<LevelLattice *> operands,
    ArrayRef<const LevelLattice *> results) {
  auto propagate = [&](Value value, const LevelState &state) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Back Propagate " << state << " to " << value << "\n");
    }
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op).Default([&](auto &op) {
    // condition on result secretness
    SmallVector<OpResult> secretResults;
    getSecretResults(&op, secretResults);
    if (secretResults.empty()) {
      return;
    }

    auto levelResult = 0;
    for (auto result : secretResults) {
      auto &levelState = getLatticeElement(result)->getValue();
      if (!levelState.isInitialized()) {
        return;
      }
      levelResult = std::max(levelResult, levelState.getLevel());
    }

    // only back-prop for non-secret operands
    SmallVector<OpOperand *> nonSecretOperands;
    getNonSecretOperands(&op, nonSecretOperands);
    for (auto *operand : nonSecretOperands) {
      propagate(operand->get(), LevelState(levelResult));
    }
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

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

/// baseLevel is for B/FV scheme, where all the analysis result would be 0
void annotateLevel(Operation *top, DataFlowSolver *solver, int baseLevel) {
  auto maxLevel = getMaxLevel(top, solver);

  auto getIntegerAttr = [&](int level) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), level);
  };

  // use L to 0 instead of 0 to L
  auto getLevel = [&](Value value) {
    return maxLevel -
           solver->lookupState<LevelLattice>(value)->getValue().getLevel() +
           baseLevel;
  };

  top->walk<WalkOrder::PreOrder>([&](mgmt::InitOp initOp) {
    auto level = getLevel(initOp.getResult());
    initOp->setAttr(kArgLevelAttrName, getIntegerAttr(level));
  });

  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    for (auto i = 0; i != genericOp.getBody()->getNumArguments(); ++i) {
      auto blockArg = genericOp.getBody()->getArgument(i);
      auto level = getLevel(blockArg);
      genericOp.setOperandAttr(i, kArgLevelAttrName, getIntegerAttr(level));
    }

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      if (!isSecret(op->getResult(0), solver)) {
        return;
      }
      auto level = getLevel(op->getResult(0));
      op->setAttr(kArgLevelAttrName, getIntegerAttr(level));
    });
  });
}

LevelState::LevelType getLevelFromMgmtAttr(Value value) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value);
  if (!mgmtAttr) {
    assert(false && "MgmtAttr not found");
  }
  return mgmtAttr.getLevel();
}

}  // namespace heir
}  // namespace mlir
