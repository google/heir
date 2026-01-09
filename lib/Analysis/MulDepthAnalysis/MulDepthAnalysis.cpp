#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"

#include <algorithm>
#include <cstdint>
#include <functional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

FailureOr<int64_t> deriveResultMulDepth(
    Operation* op, ArrayRef<const MulDepthLattice*> operands) {
  int64_t operandsMulDepth = 0;
  for (auto* operand : operands) {
    if (!operand || !operand->getValue().isInitialized()) {
      continue;
    }
    operandsMulDepth =
        std::max(operandsMulDepth, operand->getValue().getMulDepth());
  }

  int64_t increase = 0;
  if (dyn_cast<IncreasesMulDepthOpInterface>(op)) increase = 1;
  return operandsMulDepth + increase;
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
        FailureOr<int64_t> resultsMulDepth =
            deriveResultMulDepth(&op, secretOperandLattices);
        if (failed(resultsMulDepth)) {
          return;
        }

        for (auto result : secretResults) {
          propagate(result, MulDepthState(resultsMulDepth.value()));
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
