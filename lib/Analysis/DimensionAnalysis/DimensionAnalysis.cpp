#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
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

//===----------------------------------------------------------------------===//
// DimensionAnalysis (Forward)
//===----------------------------------------------------------------------===//

LogicalResult DimensionAnalysis::visitOperation(
    Operation* op, ArrayRef<const DimensionLattice*> operands,
    ArrayRef<DimensionLattice*> results) {
  auto propagate = [&](Value value, const DimensionState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation&>(*op)
      .Case<mgmt::RelinearizeOp>([&](auto relinearizeOp) {
        // implicitly ensure that the operand is secret
        propagate(relinearizeOp.getResult(), DimensionState(2));
      })
      .Default([&](auto& op) {
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

        SmallVector<OpOperand*> secretOperands;
        getSecretOperands(&op, secretOperands);
        for (auto* operand : secretOperands) {
          auto& dimensionState = getLatticeElement(operand->get())->getValue();
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
    CallOpInterface call, ArrayRef<const DimensionLattice*> argumentLattices,
    ArrayRef<DimensionLattice*> resultLattices) {
  auto callback = std::bind(&DimensionAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<DimensionState, DimensionLattice>(
      call, argumentLattices, resultLattices, callback);
}

//===----------------------------------------------------------------------===//
// DimensionAnalysis (Backward)
//===----------------------------------------------------------------------===//

void DimensionAnalysisBackward::setToExitState(DimensionLattice* lattice) {
  propagateIfChanged(lattice, lattice->join(DimensionState()));
}

LogicalResult DimensionAnalysisBackward::visitOperation(
    Operation* op, ArrayRef<DimensionLattice*> operands,
    ArrayRef<const DimensionLattice*> results) {
  auto propagate = [&](Value value, const DimensionState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  // condition on result secretness
  SmallVector<OpResult> secretResults;
  getSecretResults(op, secretResults);
  if (secretResults.empty()) {
    return success();
  }

  auto dimensionResult = 0;
  for (auto result : secretResults) {
    auto& dimensionState = getLatticeElement(result)->getValue();
    if (!dimensionState.isInitialized()) {
      return success();
    }
    dimensionResult = std::max(dimensionResult, dimensionState.getDimension());
  }

  // only back-prop for non-secret operands
  SmallVector<OpOperand*> nonSecretOperands;
  getNonSecretOperands(op, nonSecretOperands);
  for (auto* operand : nonSecretOperands) {
    propagate(operand->get(), DimensionState(dimensionResult));
  }

  // also backprop to mgmt.init if it is not secret
  for (auto& opOperand : op->getOpOperands()) {
    if (!isSecretInternal(op, opOperand.get()) &&
        isa_and_nonnull<mgmt::InitOp>(opOperand.get().getDefiningOp())) {
      propagate(opOperand.get(), DimensionState(dimensionResult));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

std::optional<DimensionState::DimensionType> getDimension(
    Value value, DataFlowSolver* solver) {
  auto* lattice = solver->lookupState<DimensionLattice>(value);
  if (!lattice) {
    return std::nullopt;
  }
  if (!lattice->getValue().isInitialized()) {
    return std::nullopt;
  }
  return lattice->getValue().getDimension();
}

int getDimensionFromMgmtAttr(Value value) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value);
  if (!mgmtAttr) {
    assert(false && "MgmtAttr not found");
  }
  return mgmtAttr.getDimension();
}

void annotateDimension(Operation* top, DataFlowSolver* solver) {
  auto getIntegerAttr = [&](int dimension) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), dimension);
  };

  auto getDimensionValue = [&](Value value) -> int {
    auto* lattice = solver->lookupState<DimensionLattice>(value);
    if (lattice && lattice->getValue().isInitialized()) {
      return lattice->getValue().getDimension();
    }
    // If the value is not initialized, try to get it from the mgmt attr.
    if (auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value)) {
      return mgmtAttr.getDimension();
    }
    return 2;
  };

  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      int dimension = getDimensionValue(value);
      setAttributeAssociatedWith(value, kArgDimensionAttrName,
                                 getIntegerAttr(dimension));
    }
  });
}

}  // namespace heir
}  // namespace mlir
