#include "lib/Analysis/LayoutFoldingAnalysis/LayoutFoldingAnalysis.h"

#include <set>

#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "layout-folding-analysis"

namespace mlir {
namespace heir {

namespace {

const std::set<std::string> layoutChangingOps = {
    "linalg.reduce",
};

const std::set<std::string> layoutFixingOps = {
    "tensor_ext.assign_layout",
};

}  // namespace

void LayoutIsFreeAnalysis::setToEntryState(LayoutIsFreeLattice* lattice) {
  // TODO(#1311): if the anchor is an arg of a non-main func, it may not be
  // free.
  propagateIfChanged(lattice, lattice->join(LayoutIsFree(true)));
}

LogicalResult isKnownToBeFree(Operation* operation,
                              ArrayRef<const LayoutIsFreeLattice*> operands) {
  if (operands.empty() ||
      layoutFixingOps.contains(operation->getName().getStringRef().str())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Op " << operation->getName() << " is known to be free.\n");
    return success();
  }
  return failure();
}

void LayoutIsFreeAnalysis::propagateToResults(
    LayoutIsFree resultLayoutIsFree, ArrayRef<LayoutIsFreeLattice*> results) {
  for (LayoutIsFreeLattice* result : results) {
    propagateIfChanged(result, result->join(resultLayoutIsFree));
  }
}

LogicalResult LayoutIsFreeAnalysis::visitOperation(
    Operation* operation, ArrayRef<const LayoutIsFreeLattice*> operands,
    ArrayRef<LayoutIsFreeLattice*> results) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting op: " << operation->getName() << "\n");

  if (succeeded(isKnownToBeFree(operation, operands))) {
    propagateToResults(LayoutIsFree(true), results);
    return success();
  }

  // If any operand is not free, then no result is free.
  LayoutIsFree resultLayoutIsFree = LayoutIsFree(true);
  for (const LayoutIsFreeLattice* operand : operands) {
    const LayoutIsFree operandLayoutIsFree = operand->getValue();
    LLVM_DEBUG(llvm::dbgs() << "Op operand is free: "
                            << operandLayoutIsFree.getValue() << "\n");
    resultLayoutIsFree =
        LayoutIsFree::join(resultLayoutIsFree, operandLayoutIsFree);
    if (!resultLayoutIsFree.getValue()) break;
  }

  // Distinguished ops that change the layout of their results.
  std::string opName(operation->getName().getStringRef());
  if (layoutChangingOps.contains(opName)) {
    LLVM_DEBUG(llvm::dbgs() << "Op is designated to make results not free\n");
    resultLayoutIsFree = LayoutIsFree(false);
  }

  propagateToResults(resultLayoutIsFree, results);
  return success();
}

void LayoutIsFreeAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const LayoutIsFreeLattice*> argumentLattices,
    ArrayRef<LayoutIsFreeLattice*> resultLattices) {
  // Calling an external function implies anything can happen, and so we can't
  // infer anything about whether the layout of the results can be changed for
  // free. We'd need some inter-procedural analysis to trace back to the
  // original assign_layout or input function argument to do better.
  for (LayoutIsFreeLattice* result : resultLattices) {
    propagateIfChanged(result, result->join(LayoutIsFree(false)));
  }
}

bool isLayoutFree(Value value, DataFlowSolver* solver) {
  auto* lattice = solver->lookupState<LayoutIsFreeLattice>(value);
  return lattice && lattice->getValue().getValue();
}

inline raw_ostream& operator<<(raw_ostream& os, const LayoutIsFree& value) {
  value.print(os);
  return os;
}

}  // namespace heir
}  // namespace mlir
