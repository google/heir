#include "lib/Analysis/LevelAnalysis/BootstrapWaterlineAnalysis.h"

#include <cassert>
#include <functional>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "bootstrap-waterline-analysis"

namespace mlir {
namespace heir {

LogicalResult BootstrapWaterlineAnalysis::visitOperation(
    Operation* op, ArrayRef<const BootstrapWaterlineLattice*> operands,
    ArrayRef<BootstrapWaterlineLattice*> results) {
  auto propagate = [&](Value value, const BootstrapWaterlineState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  // 1. Extract LevelState from operands
  SmallVector<LevelState> operandLevelStates;
  for (auto* operand : operands) {
    operandLevelStates.push_back(operand->getValue().getLevelState());
  }

  // 2. Compute prospective level
  LevelState prospectiveLevel = deriveResultLevel(op, operandLevelStates);
  if (levelBudget > 0 && prospectiveLevel.isInt() &&
      prospectiveLevel.getInt() > levelBudget) {
    prospectiveLevel = LevelState(Invalid{});
  }

  // 3. Determine if we need to reset/bootstrap
  LevelState resultLevel;
  bool resultNeedsBootstrap = false;

  if (isa<ResetsLevelOpInterface>(op)) {
    resultLevel = prospectiveLevel;
    resultNeedsBootstrap = false;
  } else {
    // We wait as long as possible to bootstrap, meaning that the level will
    // remain at the waterline (i.e., level zero) until it hits a level-reducing
    // op, at which point we have to mark the _operand_ as needing a bootstrap.
    // But since that operand has already been processed by the analysis, we
    // mark the op result and then patch it up by the pass that uses this
    // analysis.
    bool exceedsWaterline =
        prospectiveLevel.isInvalid() ||
        (prospectiveLevel.isInt() && prospectiveLevel.getInt() > waterline);

    resultNeedsBootstrap = exceedsWaterline;
    if (exceedsWaterline) {
      if (auto reduceOp = dyn_cast<ReducesLevelOpInterface>(op)) {
        resultLevel = LevelState(reduceOp.getLevelsToDrop());
      } else {
        resultLevel = LevelState(0);
      }
    } else {
      resultLevel = prospectiveLevel;
    }
  }

  BootstrapWaterlineState resultState(resultLevel, resultNeedsBootstrap);

  LLVM_DEBUG({
    llvm::dbgs() << "BWAnalysis: " << op->getName() << " prospective=";
    prospectiveLevel.print(llvm::dbgs());
    llvm::dbgs() << " -> result=";
    resultState.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  for (auto result : op->getOpResults()) {
    if (isa<mgmt::InitOp>(op) || isSecretInternal(op, result)) {
      propagate(result, resultState);
    }
  }

  return success();
}

void BootstrapWaterlineAnalysis::visitExternalCall(
    CallOpInterface call,
    ArrayRef<const BootstrapWaterlineLattice*> argumentLattices,
    ArrayRef<BootstrapWaterlineLattice*> resultLattices) {
  auto callback =
      std::bind(&BootstrapWaterlineAnalysis::propagateIfChangedWrapper, this,
                std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<BootstrapWaterlineState,
                                  BootstrapWaterlineLattice>(
      call, argumentLattices, resultLattices, callback);
}

}  // namespace heir
}  // namespace mlir
