#include "lib/Analysis/CKKSRangeAnalysis/CKKSRangeAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "CKKSRangeAnalysis"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// CKKSRangeAnalysis (Forward)
//===----------------------------------------------------------------------===//

LogicalResult CKKSRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const CKKSRangeLattice *> operands,
    ArrayRef<CKKSRangeLattice *> results) {
  auto propagate = [&](Value value, const CKKSRangeState &state) {
    auto *lattice = getLatticeElement(value);
    LLVM_DEBUG(llvm::dbgs() << "Propagate CKKSRangeState to " << value << ": "
                            << state << "\n");
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<arith::AddFOp, arith::AddIOp>([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        auto rangeResult = Log2Arithmetic::of(0);

        for (auto &operand : op->getOpOperands()) {
          auto &rangeState = getLatticeElement(operand.get())->getValue();
          if (!rangeState.isInitialized()) {
            return;
          }
          rangeResult = rangeResult + rangeState.getCKKSRange();
        }

        for (auto result : secretResults) {
          propagate(result, CKKSRangeState(rangeResult));
        }
      })
      .Case<arith::MulFOp, arith::MulIOp>([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        auto rangeResult = Log2Arithmetic::of(1);

        for (auto &operand : op->getOpOperands()) {
          auto &rangeState = getLatticeElement(operand.get())->getValue();
          if (!rangeState.isInitialized()) {
            return;
          }
          rangeResult = rangeResult * rangeState.getCKKSRange();
        }

        for (auto result : secretResults) {
          propagate(result, CKKSRangeState(rangeResult));
        }
      })
      .Case<arith::ConstantOp>([&](auto &op) {
        // For constant, the range is [constant]
        std::optional<Log2Arithmetic> range = std::nullopt;
        TypedAttr constAttr = op.getValue();
        llvm::TypeSwitch<Attribute>(constAttr)
            .Case<FloatAttr>([&](FloatAttr value) {
              range = Log2Arithmetic::of(std::fabs(value.getValueAsDouble()));
            })
            .template Case<IntegerAttr>([&](IntegerAttr value) {
              range =
                  Log2Arithmetic::of(std::abs(value.getValue().getSExtValue()));
            })
            .template Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
              auto elementType = getElementTypeOrSelf(constAttr.getType());
              if (mlir::isa<FloatType>(elementType)) {
                std::optional<APFloat> maxValue;
                for (APFloat value : denseAttr.template getValues<APFloat>()) {
                  value.clearSign();
                  if (!maxValue.has_value() ||
                      maxValue->compare(value) == APFloat::cmpLessThan) {
                    maxValue = value;
                  }
                }
                if (maxValue.has_value()) {
                  range =
                      Log2Arithmetic::of(maxValue.value().convertToDouble());
                }
              } else if (mlir::isa<IntegerType>(elementType)) {
                std::optional<APInt> maxValue;
                for (APInt value : mlir::cast<DenseElementsAttr>(constAttr)
                                       .template getValues<APInt>()) {
                  value.clearSignBit();
                  if (!maxValue.has_value() || maxValue->ule(value)) {
                    maxValue = value;
                  }
                }
                range = Log2Arithmetic::of(maxValue.value().getSExtValue());
              }
            });
        // We can encounter DenseResourceElementsAttr, we do not know its range
        if (!range.has_value()) {
          return;
        }
        for (auto result : op->getResults()) {
          propagate(result, CKKSRangeState(range.value()));
        }
      })
      .Case<mgmt::InitOp>([&](auto &op) {
        auto inputState = getLatticeElement(op->getOperand(0))->getValue();
        if (!inputState.isInitialized()) {
          return;
        }
        // For InitOp, the range is the same as the input range
        propagate(op->getResult(0), inputState);
      })
      .Case<tensor::InsertOp>([&](tensor::InsertOp op) {
        auto scalarState = getLatticeElement(op.getScalar())->getValue();
        auto destState = getLatticeElement(op.getDest())->getValue();
        if (!scalarState.isInitialized() || !destState.isInitialized()) {
          return;
        }
        auto resultState =
            CKKSRangeState::join(scalarState, destState);  // Join the ranges
        propagate(op.getResult(), resultState);
      })
      // Rotation does not change the CKKS range
      .Default([&](auto &op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(&op, secretOperands);
        if (secretOperands.empty()) {
          return;
        }

        // short-circuit to get range
        CKKSRangeState rangeState;
        for (auto *operand : secretOperands) {
          auto &operandRangeState =
              getLatticeElement(operand->get())->getValue();
          if (operandRangeState.isInitialized()) {
            rangeState = operandRangeState;
            break;
          }
        }

        for (auto result : secretResults) {
          propagate(result, rangeState);
        }
      });
  return success();
}

void CKKSRangeAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const CKKSRangeLattice *> argumentLattices,
    ArrayRef<CKKSRangeLattice *> resultLattices) {
  auto callback = std::bind(&CKKSRangeAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<CKKSRangeState, CKKSRangeLattice>(
      call, argumentLattices, resultLattices, callback);
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

std::optional<CKKSRangeState::CKKSRangeType> getCKKSRange(
    Value value, DataFlowSolver *solver) {
  auto *lattice = solver->lookupState<CKKSRangeLattice>(value);
  if (!lattice) {
    return std::nullopt;
  }
  if (!lattice->getValue().isInitialized()) {
    return std::nullopt;
  }
  return lattice->getValue().getCKKSRange();
}

}  // namespace heir
}  // namespace mlir
