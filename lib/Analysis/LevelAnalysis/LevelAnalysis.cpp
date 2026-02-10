#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/HEIRInterfaces.h"
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

#define DEBUG_TYPE "level-analysis"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// LevelAnalysis (Forward)
//===----------------------------------------------------------------------===//
static void debugLog(StringRef opName, ArrayRef<const LevelLattice*> operands,
                     const LevelState& result) {
  LLVM_DEBUG({
    llvm::dbgs() << "transferForward: " << opName << "(";
    for (auto* operand : operands) {
      operand->getValue().print(llvm::dbgs());
      llvm::dbgs() << ", ";
    }
    llvm::dbgs() << ") = ";
    result.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
};

LevelState transferForward(ReducesLevelOpInterface op,
                           ArrayRef<const LevelLattice*> operands) {
  LevelState result = std::visit(
      Overloaded{
          [](MaxLevel) -> LevelState { return LevelState(Invalid{}); },
          [](Uninit) -> LevelState { return LevelState(Invalid{}); },
          [](Invalid) -> LevelState { return LevelState(Invalid{}); },
          [&](int val) -> LevelState {
            return LevelState(val + op.getLevelsToDrop());
          },
      },
      operands[0]->getValue().get());
  LLVM_DEBUG(debugLog("ReduceLevelOpInterface", operands, result));
  return result;
}

LevelState transferForward(ReducesAllLevelsOpInterface op,
                           ArrayRef<const LevelLattice*> operands) {
  LevelState result = std::visit(
      Overloaded{
          // MaxLevel -> MaxLevel should result in a no-op, so technically
          // acceptable.
          [](MaxLevel) -> LevelState { return LevelState(MaxLevel{}); },
          [](Uninit) -> LevelState { return LevelState(Invalid{}); },
          [](Invalid) -> LevelState { return LevelState(Invalid{}); },
          [](int val) -> LevelState { return LevelState(MaxLevel{}); },
      },
      operands[0]->getValue().get());
  LLVM_DEBUG(debugLog("ReduceAllLevelsOpInterface", operands, result));
  return result;
}

LevelState transferForward(ResetsLevelOpInterface op,
                           ArrayRef<const LevelLattice*> operands) {
  LevelState result = std::visit(
      Overloaded{
          [](MaxLevel) -> LevelState { return LevelState(0); },
          [](Uninit) -> LevelState { return LevelState(Invalid{}); },
          [](Invalid) -> LevelState { return LevelState(Invalid{}); },
          [](int val) -> LevelState { return LevelState(0); },
      },
      operands[0]->getValue().get());
  LLVM_DEBUG(debugLog("ResetsLevelOpInterface", operands, result));
  return result;
}

LevelState deriveResultLevel(Operation* op,
                             ArrayRef<const LevelLattice*> operands) {
  return llvm::TypeSwitch<Operation&, LevelState>(*op)
      .Case<ResetsLevelOpInterface>(
          [&](auto op) -> LevelState { return transferForward(op, operands); })
      .Case<ReducesAllLevelsOpInterface>(
          [&](auto op) -> LevelState { return transferForward(op, operands); })
      .Case<ReducesLevelOpInterface>(
          [&](auto op) -> LevelState { return transferForward(op, operands); })
      .Default([&](auto& op) -> LevelState {
        LevelState result;
        for (auto* operandState : operands) {
          result = LevelState::join(result, operandState->getValue());
        }
        LLVM_DEBUG(debugLog(op.getName().getStringRef(), operands, result));
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
static void debugLogBackwards(StringRef opName,
                              ArrayRef<const LevelLattice*> results,
                              const LevelState& operand, unsigned operandNum) {
  LLVM_DEBUG({
    llvm::dbgs() << "transferBackward: " << opName << " results(";
    for (auto* result : results) {
      result->getValue().print(llvm::dbgs());
      llvm::dbgs() << ", ";
    }
    llvm::dbgs() << ") -> operand " << operandNum << " = ";
    operand.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
};

LogicalResult LevelAnalysisBackward::visitOperation(
    Operation* op, ArrayRef<LevelLattice*> operands,
    ArrayRef<const LevelLattice*> results) {
  auto propagate = [&](Value value, const LevelState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs() << "Back Propagating (changed) " << state
                              << " to " << value << "\n");
    }
    propagateIfChanged(lattice, changed);
  };

  // Operations that cannot have a plaintext operand do not get backward
  // propagation, since this backward pass is primarily to assign a level
  // to plaintext operands of ct-pt ops so they can be encoded properly.
  auto plaintextOperandInterface = dyn_cast<PlaintextOperandInterface>(op);
  if (!plaintextOperandInterface) {
    LLVM_DEBUG(llvm::dbgs()
               << "Not back propagating for " << op->getName()
               << " because it does not implement PlaintextOperandInterface\n");
    return success();
  }

  SmallVector<unsigned> secretResultIndices;
  getSecretResultIndices(op, secretResultIndices);
  if (secretResultIndices.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Not back propagating for " << op->getName()
                            << " because no results are secret\n");
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "Back propagating for PlaintextOperandInterface "
                          << op->getName() << "\n");

  LevelState levelResult;
  for (unsigned i : secretResultIndices) {
    LevelState state = results[i]->getValue();
    if (state.isInt() || state.isMaxLevel())
      levelResult = LevelState::join(levelResult, state);
  }

  SmallVector<OpOperand*> plaintextOperands;
  getPlaintextOperands(op, plaintextOperands);
  for (auto* operand : plaintextOperands) {
    LLVM_DEBUG(debugLogBackwards(op->getName().getStringRef(), results,
                                 levelResult, operand->getOperandNumber()));
    propagate(operand->get(), levelResult);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Walk the entire IR and return the maximum assigned level of all secret
// Values.
int getMaxLevel(Operation* top, DataFlowSolver* solver) {
  auto maxLevel = 0;
  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      auto levelState = solver->lookupState<LevelLattice>(value)->getValue();
      if (levelState.isInt()) {
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
    if (levelState.isMaxLevel()) {
      return 0;  // reversing, the "max" level becomes level 0
    }
    if (!levelState.isInt()) {
      return maxLevel + baseLevel;
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
