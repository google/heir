#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult DimensionAnalysis::visitOperation(
    Operation *op, ArrayRef<const DimensionLattice *> operands,
    ArrayRef<DimensionLattice *> results) {
  auto propagate = [&](Value value, const DimensionState &state) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, DimensionState(2));
        }
      })
      .Case<mgmt::RelinearizeOp>([&](auto relinearizeOp) {
        // implicitly ensure that the operand is secret
        propagate(relinearizeOp.getResult(), DimensionState(2));
      })
      .Default([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        auto isMul = false;

        if (isa<arith::MulIOp, arith::MulFOp>(op)) {
          isMul = true;
        }

        // for mul, initialize to 0, for max, initialize to 2
        auto dimensionResult = isMul ? 0 : 2;

        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(&op, secretOperands);
        for (auto *operand : secretOperands) {
          auto &dimensionState = getLatticeElement(operand->get())->getValue();
          if (!dimensionState.isInitialized()) {
            return;
          }
          auto dimension = dimensionState.getDimension();
          if (isMul) {
            dimensionResult += dimension;
          } else {
            dimensionResult = std::max(dimensionResult, dimension);
          }
        }

        // tensor product
        if (isMul && secretOperands.size() == 2) {
          dimensionResult -= 1;
        }

        for (auto result : secretResults) {
          propagate(result, DimensionState(dimensionResult));
        }
      });
  return success();
}

int getDimension(Value value, DataFlowSolver *solver) {
  auto *lattice = solver->lookupState<DimensionLattice>(value);
  if (!lattice) {
    assert(false && "DimensionLattice not found");
    return 2;
  }
  if (!lattice->getValue().isInitialized()) {
    assert(false && "DimensionLattice not initialized");
    return 2;
  }
  return lattice->getValue().getDimension();
}

void annotateDimension(Operation *top, DataFlowSolver *solver) {
  auto getIntegerAttr = [&](int dimension) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), dimension);
  };

  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    for (auto blockArg : genericOp.getBody()->getArguments()) {
      genericOp.setArgAttr(blockArg.getArgNumber(), "dimension",
                           getIntegerAttr(getDimension(blockArg, solver)));
    }

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      if (!ensureSecretness(op->getResult(0), solver)) {
        return;
      }
      op->setAttr("dimension",
                  getIntegerAttr(getDimension(op->getResult(0), solver)));
    });
  });
}

}  // namespace heir
}  // namespace mlir
