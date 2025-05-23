#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>

#include "SchemeSelectionAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"        // from @llvm-project
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
  LLVM_DEBUG(llvm::dbgs()
      << "Visiting: " << op->getName() << ". ");

  auto propagate = [&](Value value, const NatureOfComputation &counter) {
    auto *oldNoc = getLatticeElement(value);
    ChangeResult changed = oldNoc->join(counter);
    propagateIfChanged(oldNoc, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      // count integer arithmetic ops
      .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp>([&](auto intOp) {
		this->counter = this->counter + NatureOfComputation(0, 0, 1, 0, 0, 0);
        auto newNoc = counter;
        intOp->setAttr(numIntArithOpsAttrName, IntegerAttr::get(IntegerType::get(op->getContext(), 64), newNoc.getIntArithOpsCount()));
		    LLVM_DEBUG(llvm::dbgs()
      	<< "Counting: " << newNoc << "\n");
        propagate(intOp.getResult(), newNoc);
      })
      // count real arithmetic ops
      .Case<arith::AddFOp, arith::SubFOp, arith::MulFOp>([&](auto realOp) {
		this->counter = this->counter + NatureOfComputation(0, 0, 0, 1, 0, 0);
        auto newNoc = counter;
        realOp->setAttr(numRealArithOpsAttrName, IntegerAttr::get(IntegerType::get(op->getContext(), 64), newNoc.getRealArithOpsCount()));
		LLVM_DEBUG(llvm::dbgs()
      	<< "Counting: " << newNoc << "\n");
        propagate(realOp->getResult(0), newNoc);
      })
      // count non linear ops
      .Case<math::AbsFOp, math::AbsIOp>([&](auto nonLinOp) {
        this->counter = this->counter + NatureOfComputation(0, 0, 0, 0, 0, 1);
        auto newNoc = counter;
        nonLinOp->setAttr(numNonLinOpsAttrName, IntegerAttr::get(IntegerType::get(op->getContext(), 64), newNoc.getNonLinOpsCount()));
		LLVM_DEBUG(llvm::dbgs()
      	<< "Counting: " << newNoc << "\n");
        propagate(nonLinOp->getResult(0), newNoc);
      })
      // count bool ops
      .Case<arith::AndIOp, arith::OrIOp, arith::XOrIOp>([&](auto boolOp) {
        this->counter = this->counter + NatureOfComputation(1, 0, 0, 0, 0, 0);
        auto newNoc = counter;
        boolOp->setAttr(numBoolOpsAttrName, IntegerAttr::get(IntegerType::get(op->getContext(), 64), newNoc.getBoolOpsCount()));
		LLVM_DEBUG(llvm::dbgs()
      	<< "Counting: " << newNoc << "\n");
        propagate(boolOp->getResult(0), newNoc);
      })
      // count bit ops
      .Case<arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>([&](auto bitOp) {
        this->counter = this->counter + NatureOfComputation(0, 1, 0, 0, 0, 0);
        auto newNoc = counter;
        bitOp->setAttr(numBitOpsAttrName, IntegerAttr::get(IntegerType::get(op->getContext(), 64), newNoc.getBitOpsCount()));
		LLVM_DEBUG(llvm::dbgs()
      	<< "Counting: " << newNoc << "\n");
        propagate(bitOp->getResult(0), newNoc);
      })
      // count real comparisons
      .Case<arith::CmpFOp>([&](auto cmpOps) {
        this->counter = this->counter + NatureOfComputation(0, 0, 0, 1, 1, 0);
        auto newNoc = counter;
        cmpOps->setAttr(numCmpOpsAttrName, IntegerAttr::get(IntegerType::get(op->getContext(), 64), newNoc.getCmpOpsCount()));
		LLVM_DEBUG(llvm::dbgs()
      	<< "Counting: " << newNoc << "\n");
        propagate(cmpOps->getResult(0), newNoc);
      })
	  // count int comparisons
      .Case<arith::CmpIOp>([&](auto cmpOps) {
        this->counter = this->counter + NatureOfComputation(0, 0, 1, 0, 1, 0);
        auto newNoc = counter;
        cmpOps->setAttr(numCmpOpsAttrName, IntegerAttr::get(IntegerType::get(op->getContext(), 64), newNoc.getCmpOpsCount()));
		LLVM_DEBUG(llvm::dbgs()
      	<< "Counting: " << newNoc << "\n");
        propagate(cmpOps->getResult(0), newNoc);
      });
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

static NatureOfComputation getMaxNatComp(Operation *top, DataFlowSolver *solver) {
    auto maxNatComp = NatureOfComputation(0,0,0,0,0,0);
    top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      funcOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
          LLVM_DEBUG(llvm::dbgs()
            << "Writing annotations here: " << op->getName() << "\n");
          if (op->getNumResults() == 0) {
            return;
          }
          auto natcomp = solver->lookupState<SchemeInfoLattice>(op->getResult(0))->getValue();
          if (natcomp.isInitialized()) {
            maxNatComp = NatureOfComputation::max(maxNatComp, natcomp);
          }
      });
  });
  return maxNatComp;
}

void annotateNatureOfComputation(Operation *top, DataFlowSolver *solver,
                                 int baseLevel) {

  auto getIntegerAttr = [&](int level) {
    return IntegerAttr::get(IntegerType::get(top->getContext(), 64), level);
  };

  auto getNatureOfComputationAttribute = [&](Value value) {
    return solver->lookupState<SchemeInfoLattice>(value)->getValue().getDominantAttributeName();
  };
  
  auto getNatureOfComputationCount = [&](Value value) {
    return solver->lookupState<SchemeInfoLattice>(value)->getValue().getDominantComputationCount();
  };

  auto maxNatComp = getMaxNatComp(top, solver);
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    funcOp->setAttr(numBoolOpsAttrName, getIntegerAttr(maxNatComp.getBoolOpsCount()));
    funcOp->setAttr(numBitOpsAttrName, getIntegerAttr(maxNatComp.getBitOpsCount()));
    funcOp->setAttr(numIntArithOpsAttrName, getIntegerAttr(maxNatComp.getIntArithOpsCount()));
    funcOp->setAttr(numRealArithOpsAttrName, getIntegerAttr(maxNatComp.getRealArithOpsCount()));
    funcOp->setAttr(numCmpOpsAttrName, getIntegerAttr(maxNatComp.getCmpOpsCount()));
    funcOp->setAttr(numNonLinOpsAttrName, getIntegerAttr(maxNatComp.getNonLinOpsCount()));
    LLVM_DEBUG(llvm::dbgs()
      << "Writing annotations here: " << funcOp->getName() << "\n");
  });

}

}  // namespace heir
}  // namespace mlir