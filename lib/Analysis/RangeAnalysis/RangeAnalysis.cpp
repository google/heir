#include "lib/Analysis/RangeAnalysis/RangeAnalysis.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <optional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Utils/LogArithmetic.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

#define DEBUG_TYPE "RangeAnalysis"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// RangeAnalysis (Forward)
//===----------------------------------------------------------------------===//

LogicalResult RangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const RangeLattice *> operands,
    ArrayRef<RangeLattice *> results) {
  auto propagate = [&](Value value, const RangeState &state) {
    auto *lattice = getLatticeElement(value);
    LLVM_DEBUG(llvm::dbgs()
               << "Propagate RangeState to " << value << ": " << state << "\n");
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
          rangeResult = rangeResult + rangeState.getRange();
        }

        for (auto result : secretResults) {
          propagate(result, RangeState(rangeResult));
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
          rangeResult = rangeResult * rangeState.getRange();
        }

        for (auto result : secretResults) {
          propagate(result, RangeState(rangeResult));
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
          propagate(result, RangeState(range.value()));
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
            RangeState::join(scalarState, destState);  // Join the ranges
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
        RangeState rangeState;
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

void RangeAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const RangeLattice *> argumentLattices,
    ArrayRef<RangeLattice *> resultLattices) {
  auto callback = std::bind(&RangeAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<RangeState, RangeLattice>(
      call, argumentLattices, resultLattices, callback);
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

std::optional<RangeState::RangeType> getRange(Value value,
                                              DataFlowSolver *solver) {
  auto *lattice = solver->lookupState<RangeLattice>(value);
  if (!lattice) {
    return std::nullopt;
  }
  if (!lattice->getValue().isInitialized()) {
    return std::nullopt;
  }
  return lattice->getValue().getRange();
}

}  // namespace heir
}  // namespace mlir
