#include "lib/Analysis/PreprocessingAnalysis/PreprocessingAnalysis.h"

#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

void PreprocessingAnalysis::setToExitState(PreprocessingLattice* lattice) {
  propagateIfChanged(lattice, lattice->join(PreprocessingState(true)));
}

LogicalResult PreprocessingAnalysis::visitOperation(
    Operation* op, ArrayRef<PreprocessingLattice*> operands,
    ArrayRef<const PreprocessingLattice*> results) {
  // Join all result lattices
  PreprocessingState combinedState;
  for (const PreprocessingLattice* resultLattice : results) {
    if (resultLattice) {
      combinedState =
          PreprocessingState::join(combinedState, resultLattice->getValue());
    }
  }

  if (isa<PlaintextEncodeOpInterface>(op)) {
    combinedState.insert(op);
  }

  if (!combinedState.isInitialized()) {
    return success();
  }

  for (PreprocessingLattice* operandLattice : operands) {
    if (operandLattice) {
      ChangeResult changed = operandLattice->join(combinedState);
      propagateIfChanged(operandLattice, changed);
    }
  }

  return success();
}

void PreprocessingAnalysis::visitBranchOperand(OpOperand& operand) {
  if (!operand.getOwner()) return;
  if (auto* lattice = getLatticeElement(operand.get())) {
    PreprocessingState combined;
    for (Value res : operand.getOwner()->getResults()) {
      if (auto* resLattice = getLatticeElement(res)) {
        combined = PreprocessingState::join(combined, resLattice->getValue());
      }
    }
    ChangeResult changed = lattice->join(combined);
    propagateIfChanged(lattice, changed);
  }
}

void PreprocessingAnalysis::visitNonControlFlowArguments(
    RegionSuccessor& successor, ArrayRef<BlockArgument> arguments) {
  if (arguments.empty()) return;
  Operation* parentOp = arguments[0].getOwner()->getParentOp();
  if (!parentOp) return;

  PreprocessingState combined;
  for (Value res : parentOp->getResults()) {
    if (auto* resLattice = getLatticeElement(res)) {
      combined = PreprocessingState::join(combined, resLattice->getValue());
    }
  }

  for (BlockArgument arg : arguments) {
    if (auto* lattice = getLatticeElement(arg)) {
      ChangeResult changed = lattice->join(combined);
      propagateIfChanged(lattice, changed);
    }
  }
}

}  // namespace heir
}  // namespace mlir
