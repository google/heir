#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "SchemeSelection"

namespace mlir {
namespace heir {

LogicalResult SchemeSelectionAnalysis::visitOperation(
    Operation *op, ArrayRef<const SchemeInfoLattice *> operands,
    ArrayRef<SchemeInfoLattice *> results) {

  auto propagate = [&](Value value, const std::string opType, const NatureOfComputation &counter) {
    LLVM_DEBUG(llvm::dbgs()
      << "Visiting " << opType << ": " << op->getName() << "\n");

    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(counter);
    propagateIfChanged(lattice, changed);
  };

  int executedOpsCount = 0;  // Counter for executed integer operations

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, "no op", NatureOfComputation());
        }

        // Walk through the operations in the body to count executed integer operations
        body->walk([&](Operation *innerOp) {
          if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(innerOp)) {
            executedOpsCount++;
          }
        });
      })
      .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp>([&](auto intOp) {
        NatureOfComputation intArithOpCount(0, 0, 1, 0, 0, 0);
        propagate(intOp->getResult(0), "int op", intArithOpCount);
        executedOpsCount++;  // Increment the count for integer operations
        // Annotate the operation with execution count
        intOp->setAttr("exec_count", IntegerAttr::get(IntegerType::get(op->getContext(), 32), 1));
      })
      .Case<arith::AddFOp, arith::SubFOp, arith::MulFOp>([&](auto realOp) {
        NatureOfComputation realArithOpCount(0, 0, 0, 1, 0, 0);
        propagate(realOp->getResult(0), "real op", realArithOpCount);
      })
      .Default([&](Operation &otherOp) {
        propagate(otherOp.getResult(0), "no op", NatureOfComputation());
    });

  // If this operation is a function, annotate it with the total count
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    funcOp->setAttr("executed_integer_operations", IntegerAttr::get(IntegerType::get(op->getContext(), 32), executedOpsCount));
  }

  return success();
}

bool hasAtLeastOneBooleanOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (operand.getType().isInteger(1)) {
      return true;
    }
  }
  return false;
}

bool hasAtLeastOneIntegerOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (operand.getType().isInteger()) {
      return true;
    }
  }
  return false;
}

bool hasAtLeastOneRealOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (operand.getType().isF16() || operand.getType().isF32() ||
        operand.getType().isF64() || operand.getType().isF128()) {
      return true;
    }
  }
  return false;
}

void annotateNatureOfComputation(Operation *top, DataFlowSolver *solver,
                                 int baseLevel) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
 	  LLVM_DEBUG(llvm::dbgs()
      << "Walking here: " << genericOp->getName() << "\n");

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
	  LLVM_DEBUG(llvm::dbgs()
      << "Walking the inner loop: " << op->getName() << "\n");

      if (op->getNumResults() == 0) {
        return;
      }
      if (!isSecret(op->getResult(0), solver)) {
        return;
      }

      SmallVector<Attribute, 4> natOfCompValues;

      if (hasAtLeastOneBooleanOperand(op)) {
        natOfCompValues.push_back(StringAttr::get(top->getContext(), "bool"));
      }
      if (hasAtLeastOneIntegerOperand(op)) {
        natOfCompValues.push_back(StringAttr::get(top->getContext(), "int"));
      }
      if (hasAtLeastOneRealOperand(op)) {
        natOfCompValues.push_back(StringAttr::get(top->getContext(), "real"));
      }

      if (!natOfCompValues.empty()) {
        auto natOfCompAttr = ArrayAttr::get(top->getContext(), natOfCompValues);
        op->setAttr("natOfComp", natOfCompAttr);
      }
    });
  });
}

}  // namespace heir
}  // namespace mlir