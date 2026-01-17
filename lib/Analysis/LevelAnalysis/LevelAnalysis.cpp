#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
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

LevelState transferForward(mgmt::ModReduceOp op,
                           ArrayRef<const LevelLattice*> operands) {
  return std::visit(
      Overloaded{
          [](MaxLevel) -> LevelState { return LevelState(Invalid{}); },
          [](Uninit) -> LevelState { return LevelState(Invalid{}); },
          [](Invalid) -> LevelState { return LevelState(Invalid{}); },
          [](int val) -> LevelState { return LevelState(val + 1); },
      },
      operands[0]->getValue().get());
}

LevelState transferForward(mgmt::LevelReduceOp op,
                           ArrayRef<const LevelLattice*> operands) {
  return std::visit(
      Overloaded{
          [](MaxLevel) -> LevelState { return LevelState(Invalid{}); },
          [](Uninit) -> LevelState { return LevelState(Invalid{}); },
          [](Invalid) -> LevelState { return LevelState(Invalid{}); },
          [&](int val) -> LevelState {
            return LevelState(val + (int)op.getLevelToDrop());
          },
      },
      operands[0]->getValue().get());
}

LevelState transferForward(mgmt::LevelReduceMinOp op,
                           ArrayRef<const LevelLattice*> operands) {
  return std::visit(
      Overloaded{
          // MaxLevel -> MaxLevel should result in a no-op, so technically
          // acceptable.
          [](MaxLevel) -> LevelState { return LevelState(MaxLevel{}); },
          [](Uninit) -> LevelState { return LevelState(Invalid{}); },
          [](Invalid) -> LevelState { return LevelState(Invalid{}); },
          [](int val) -> LevelState { return LevelState(MaxLevel{}); },
      },
      operands[0]->getValue().get());
}

LevelState transferForward(mgmt::BootstrapOp op,
                           ArrayRef<const LevelLattice*> operands) {
  return std::visit(
      Overloaded{
          [](MaxLevel) -> LevelState { return LevelState(0); },
          [](Uninit) -> LevelState { return LevelState(Invalid{}); },
          [](Invalid) -> LevelState { return LevelState(Invalid{}); },
          [](int val) -> LevelState { return LevelState(0); },
      },
      operands[0]->getValue().get());
}

LevelState deriveResultLevel(Operation* op,
                             ArrayRef<const LevelLattice*> operands) {
  return llvm::TypeSwitch<Operation&, LevelState>(*op)
      .Case<mgmt::ModReduceOp, mgmt::LevelReduceOp, mgmt::BootstrapOp,
            mgmt::LevelReduceMinOp>(
          [&](auto op) -> LevelState { return transferForward(op, operands); })
      .Default([&](auto& op) -> LevelState {
        LevelState result;
        for (auto* operandState : operands) {
          result = LevelState::join(result, operandState->getValue());
        }
        return result;
      });
}

LogicalResult LevelAnalysis::visitOperation(
    Operation* op, ArrayRef<const LevelLattice*> operands,
    ArrayRef<LevelLattice*> results) {
  auto propagate = [&](Value value, const LevelState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  LLVM_DEBUG(llvm::dbgs() << "Forward Propagate visiting " << op->getName()
                          << "\n");

  LevelState resultLevel = deriveResultLevel(op, operands);
  SmallVector<OpResult> secretResults;
  getSecretResults(op, secretResults);
  for (auto result : secretResults) {
    propagate(result, resultLevel);
  }

  return success();
}

void LevelAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const LevelLattice*> argumentLattices,
    ArrayRef<LevelLattice*> resultLattices) {
  auto callback = std::bind(&LevelAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<LevelState, LevelLattice>(
      call, argumentLattices, resultLattices, callback);
}

//===----------------------------------------------------------------------===//
// LevelAnalysis (Backward)
//===----------------------------------------------------------------------===//

// The backward propagation is only meant to propagate the level required for a
// plaintext operand, which depends on the required level of the op that uses
// it. For cases where a plaintext operand is reused for multiple ct-pt ops with
// different levels, this will assign the join (max) of the levels, which can be
// managed with additional level_reduce ops; however, other passes can also
// insert additional mgmt.init ops (one for each use) to provide separate SSA
// values that will correspond later to separate packing ops.
LogicalResult LevelAnalysisBackward::visitOperation(
    Operation* op, ArrayRef<LevelLattice*> operands,
    ArrayRef<const LevelLattice*> results) {
  auto propagate = [&](Value value, const LevelState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Back Propagate " << state << " to " << value << "\n");
    }
    propagateIfChanged(lattice, changed);
  };

  LevelState levelResult;
  for (auto* resultLattice : results) {
    levelResult = LevelState::join(levelResult, resultLattice->getValue());
  }

  // only back-prop for non-secret operands
  SmallVector<OpOperand*> nonSecretOperands;
  getNonSecretOperands(op, nonSecretOperands);
  for (auto* operand : nonSecretOperands) {
    propagate(operand->get(), levelResult);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Walk the entire IR and return the maximum assigned level of all secret
// Values.
static int getMaxLevel(Operation* top, DataFlowSolver* solver) {
  auto maxLevel = 0;
  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      auto levelState = solver->lookupState<LevelLattice>(value)->getValue();
      if (levelState.isInitialized()) {
        int level = levelState.getInt();
        maxLevel = std::max(maxLevel, level);
      }
    }
  });
  return maxLevel;
}

/// baseLevel is for B/FV scheme, where all the analysis result would be 0
void annotateLevel(Operation* top, DataFlowSolver* solver, int baseLevel) {
  auto maxLevel = getMaxLevel(top, solver);

  auto getIntegerAttr = [&](int level) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), level);
  };

  // use L to 0 instead of 0 to L
  auto getLevel = [&](Value value) -> int {
    LevelState levelState =
        solver->lookupState<LevelLattice>(value)->getValue();
    // The analysis uses 0 to L for ease of analysis, and then we materialize
    // it in the IR in reverse, from L to 0.
    if (!levelState.isInitialized()) {
      return maxLevel + baseLevel;
    }
    if (levelState.isMaxLevel()) {
      return 0;  // reversing, the max level becomes level 0
    }
    return maxLevel - levelState.getInt() + baseLevel;
  };

  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      int level = getLevel(value);
      setAttributeAssociatedWith(value, kArgLevelAttrName,
                                 getIntegerAttr(level));
    }
  });
}

LevelState getLevelFromMgmtAttr(Value value) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value);
  if (!mgmtAttr) {
    assert(false && "MgmtAttr not found");
  }
  return mgmtAttr.getLevel();
}

std::optional<int> getMaxLevel(Operation* root) {
  int maxLevel = 0;
  root->walk([&](func::FuncOp funcOp) {
    if (isClientHelper(funcOp)) {
      return;
    }

    for (BlockArgument arg : funcOp.getArguments()) {
      if (isa<secret::SecretType>(arg.getType())) {
        maxLevel = std::max(maxLevel, (int)getLevelFromMgmtAttr(arg).getInt());
      }
    }
  });
  return maxLevel;
}

}  // namespace heir
}  // namespace mlir
