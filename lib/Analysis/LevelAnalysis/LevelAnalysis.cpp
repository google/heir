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

FailureOr<int64_t> deriveResultLevel(Operation* op,
                                     ArrayRef<const LevelLattice*> operands) {
  return llvm::TypeSwitch<Operation&, FailureOr<int64_t>>(*op)
      .Case<mgmt::ModReduceOp>([&](auto modReduceOp) -> FailureOr<int64_t> {
        // implicitly ensure that the operand is secret
        const auto* operandLattice = operands[0];
        if (!operandLattice->getValue().isInitialized()) {
          return failure();
        }
        int64_t level = operandLattice->getValue().getLevel();
        return level + 1;
      })
      .Case<mgmt::LevelReduceOp>([&](auto levelReduceOp) -> FailureOr<int64_t> {
        // implicitly ensure that the operand is secret
        const auto* operandLattice = operands[0];
        if (!operandLattice->getValue().isInitialized()) {
          return failure();
        }
        return operandLattice->getValue().getLevel() +
               levelReduceOp.getLevelToDrop();
      })
      .Case<mgmt::BootstrapOp>([&](auto bootstrapOp) -> FailureOr<int64_t> {
        // implicitly ensure that the result is secret
        // reset level to 0
        // TODO(#1207): reset level to currentLevel - bootstrapDepth
        return 0;
      })
      .Default([&](auto& op) -> FailureOr<int64_t> {
        auto levelResult = 0;
        for (auto* levelState : operands) {
          if (!levelState || !levelState->getValue().isInitialized()) {
            continue;
          }
          levelResult =
              std::max(levelResult, levelState->getValue().getLevel());
        }

        return levelResult;
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

  SmallVector<OpOperand*> secretOperands;
  getSecretOperands(op, secretOperands);
  SmallVector<const LevelLattice*, 2> secretOperandLattices;
  for (auto* operand : secretOperands) {
    secretOperandLattices.push_back(getLatticeElement(operand->get()));
  }
  FailureOr<int64_t> resultLevel = deriveResultLevel(op, secretOperandLattices);
  if (failed(resultLevel)) {
    // Ignore failure and continue
    return success();
  }

  SmallVector<OpResult> secretResults;
  getSecretResults(op, secretResults);
  for (auto result : secretResults) {
    propagate(result, LevelState(resultLevel.value()));
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

  llvm::TypeSwitch<Operation&>(*op).Default([&](auto& op) {
    // condition on result secretness
    SmallVector<OpResult> secretResults;
    getSecretResults(&op, secretResults);
    if (secretResults.empty()) {
      return;
    }

    auto levelResult = 0;
    for (auto result : secretResults) {
      auto& levelState = getLatticeElement(result)->getValue();
      if (!levelState.isInitialized()) {
        return;
      }
      levelResult = std::max(levelResult, levelState.getLevel());
    }

    // only back-prop for non-secret operands
    SmallVector<OpOperand*> nonSecretOperands;
    getNonSecretOperands(&op, nonSecretOperands);
    for (auto* operand : nonSecretOperands) {
      propagate(operand->get(), LevelState(levelResult));
    }
  });
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
        auto level = levelState.getLevel();
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
    return maxLevel - levelState.getLevel() + baseLevel;
  };

  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      int level = getLevel(value);
      setAttributeAssociatedWith(value, kArgLevelAttrName,
                                 getIntegerAttr(level));
    }
  });
}

LevelState::LevelType getLevelFromMgmtAttr(Value value) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value);
  if (!mgmtAttr) {
    assert(false && "MgmtAttr not found");
  }
  return mgmtAttr.getLevel();
}

std::optional<int> getMaxLevel(Operation* root) {
  int maxLevel = 0;
  root->walk([&](func::FuncOp funcOp) {
    // Skip client helpers
    if (isClientHelper(funcOp)) {
      return;
    }

    for (BlockArgument arg : funcOp.getArguments()) {
      if (isa<secret::SecretType>(arg.getType())) {
        maxLevel = std::max(maxLevel, getLevelFromMgmtAttr(arg));
      }
    }
  });
  return maxLevel;
}

}  // namespace heir
}  // namespace mlir
