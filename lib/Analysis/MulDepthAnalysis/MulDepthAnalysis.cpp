#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"

#include <algorithm>
#include <cstdint>
#include <functional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "muldepth-analysis"

namespace mlir {
namespace heir {

static void debugLog(StringRef opName,
                     ArrayRef<const MulDepthLattice*> operands,
                     const MulDepthState& result) {
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

MulDepthState deriveResultMulDepth(Operation* op,
                                   ArrayRef<const MulDepthLattice*> operands) {
  if (isa<ResetsMulDepthOpInterface>(op)) return MulDepthState(0);

  MulDepthState resultState(0);
  for (auto* operand : operands) {
    if (!operand || !operand->getValue().isInitialized()) {
      continue;
    }
    resultState = MulDepthState::join(resultState, operand->getValue());
  }

  if (resultState.isInvalid()) return resultState;

  int64_t operandsMulDepth = resultState.getMulDepth();
  int64_t increase = 0;
  if (dyn_cast<IncreasesMulDepthOpInterface>(op)) increase = 1;
  return MulDepthState(operandsMulDepth + increase);
}

LogicalResult MulDepthAnalysis::visitOperation(
    Operation* op, ArrayRef<const MulDepthLattice*> operands,
    ArrayRef<MulDepthLattice*> results) {
  auto propagate = [&](Value value, const MulDepthState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation&>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block* body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, MulDepthState(0));
        }
      })
      .Default([&](auto& op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        SmallVector<OpOperand*> secretOperands;
        getSecretOperands(&op, secretOperands);
        SmallVector<const MulDepthLattice*, 2> secretOperandLattices;
        for (auto* operand : secretOperands) {
          secretOperandLattices.push_back(getLatticeElement(operand->get()));
        }
        MulDepthState resultState =
            deriveResultMulDepth(&op, secretOperandLattices);
        if (resultState.isInt() && resultState.getMulDepth() > mulDepthBudget) {
          resultState = MulDepthState(MulDepthState::Invalid{});
        }
        debugLog(op.getName().getStringRef(), operands, resultState);
        for (auto result : secretResults) {
          propagate(result, resultState);
        }
      });
  return success();
}

void MulDepthAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const MulDepthLattice*> argumentLattices,
    ArrayRef<MulDepthLattice*> resultLattices) {
  auto callback = std::bind(&MulDepthAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<MulDepthState, MulDepthLattice>(
      call, argumentLattices, resultLattices, callback);
}

int64_t getMaxMulDepth(Operation* op, DataFlowSolver& solver) {
  int64_t maxMulDepth = 0;
  op->walk([&](Operation* op) {
    if (op->getNumResults() == 0) {
      return;
    }
    // the first result suffices as all results share the same mulDepth.
    auto* lattice = solver.lookupState<MulDepthLattice>(op->getResult(0));
    if (!lattice) {
      return;
    }
    auto& state = lattice->getValue();
    if (!state.isInitialized()) {
      return;
    }
    maxMulDepth = std::max(maxMulDepth, state.getMulDepth());
  });
  return maxMulDepth;
}

}  // namespace heir
}  // namespace mlir
