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

  auto ensureSecretness = [&](Operation *op, Value value) -> bool {
    // create dependency on SecretnessAnalysis
    auto *lattice =
        getOrCreateFor<SecretnessLattice>(getProgramPointAfter(op), value);
    if (!lattice->getValue().isInitialized()) {
      return false;
    }
    return lattice->getValue().getSecretness();
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
        if (op.getNumResults() == 0) {
          return;
        }

        // condition on result secretness
        auto secretness = ensureSecretness(&op, op.getResult(0));
        if (!secretness) {
          return;
        }

        auto isMul = false;
        if (isa<arith::MulIOp, arith::MulFOp>(op)) {
          isMul = true;
        }

        auto dimensionResult = 0;
        auto operandSecretNum = 0;
        for (const auto *operand : operands) {
          auto secretness = ensureSecretness(&op, operand->getAnchor());
          // pt/ct default
          auto dimension = 2;
          bool operandIsSecret = false;
          if (secretness) {
            if (!operand->getValue().isInitialized()) {
              return;
            }
            // ct
            operandIsSecret = true;
            operandSecretNum += 1;
            dimension = operand->getValue().getDimension();
          }

          if (isMul && operandIsSecret) {
            dimensionResult += dimension;
          } else {
            dimensionResult = std::max(dimensionResult, dimension);
          }
        }
        // tensor product
        if (isMul && operandSecretNum == 2) {
          dimensionResult -= 1;
        }

        for (auto result : op.getResults()) {
          propagate(result, DimensionState(dimensionResult));
        }
      });
  return success();
}

void annotateDimension(Operation *top, DataFlowSolver *solver) {
  auto getIntegerAttr = [&](int dimension) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), dimension);
  };

  auto getDimension = [&](Value value) {
    return solver->lookupState<DimensionLattice>(value)
        ->getValue()
        .getDimension();
  };

  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    for (auto blockArg : genericOp.getBody()->getArguments()) {
      genericOp.setArgAttr(blockArg.getArgNumber(), "dimension",
                           getIntegerAttr(getDimension(blockArg)));
    }

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      op->setAttr("dimension", getIntegerAttr(getDimension(op->getResult(0))));
    });
  });
}

}  // namespace heir
}  // namespace mlir
