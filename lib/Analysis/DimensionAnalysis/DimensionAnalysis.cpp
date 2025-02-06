#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"

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
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
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

void DimensionAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const DimensionLattice *> argumentLattices,
    ArrayRef<DimensionLattice *> resultLattices) {
  auto callback = std::bind(&DimensionAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<DimensionState, DimensionLattice>(
      call, argumentLattices, resultLattices, callback);
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

int getDimensionFromMgmtAttr(Value value) {
  Attribute attr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto genericOp = dyn_cast<secret::GenericOp>(parentOp);
    if (genericOp) {
      attr = genericOp.getOperandAttr(blockArg.getArgNumber(),
                                      mgmt::MgmtDialect::kArgMgmtAttrName);
    }
  } else {
    auto *parentOp = value.getDefiningOp();
    attr = parentOp->getAttr(mgmt::MgmtDialect::kArgMgmtAttrName);
  }
  if (!mlir::isa<mgmt::MgmtAttr>(attr)) {
    assert(false && "MgmtAttr not found");
  }
  auto mgmtAttr = mlir::cast<mgmt::MgmtAttr>(attr);
  return mgmtAttr.getDimension();
}

void annotateDimension(Operation *top, DataFlowSolver *solver) {
  auto getIntegerAttr = [&](int dimension) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), dimension);
  };

  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    for (auto blockArg : genericOp.getBody()->getArguments()) {
      genericOp.setOperandAttr(blockArg.getArgNumber(), "dimension",
                               getIntegerAttr(getDimension(blockArg, solver)));
    }

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      if (!isSecret(op->getResult(0), solver)) {
        return;
      }
      op->setAttr("dimension",
                  getIntegerAttr(getDimension(op->getResult(0), solver)));
    });
  });
}

}  // namespace heir
}  // namespace mlir
