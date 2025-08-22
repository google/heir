#include "lib/Analysis/LayoutFoldingAnalysis/LayoutFoldingAnalysis.h"

#include <set>

#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

namespace {

const std::set<std::string> layoutChangingOps = {
    "linalg.reduce",
};

}  // namespace

void LayoutIsFreeAnalysis::setToEntryState(LayoutIsFreeLattice* lattice) {
  // TODO(#1311): if the anchor is an arg of a non-main func, it may not be
  // free.
  propagateIfChanged(lattice, lattice->join(LayoutIsFree(true)));
}

LogicalResult LayoutIsFreeAnalysis::visitOperation(
    Operation* operation, ArrayRef<const LayoutIsFreeLattice*> operands,
    ArrayRef<LayoutIsFreeLattice*> results) {
  LayoutIsFree resultLayoutIsFree = LayoutIsFree(true);

  // No operands -> compiler-generated
  if (operands.empty()) {
    resultLayoutIsFree = LayoutIsFree(true);
  }

  // If any operand is not free, then no result is free.
  for (const LayoutIsFreeLattice* operand : operands) {
    const LayoutIsFree operandLayoutIsFree = operand->getValue();
    resultLayoutIsFree =
        LayoutIsFree::join(resultLayoutIsFree, operandLayoutIsFree);
    if (!resultLayoutIsFree.getValue()) break;
  }

  // Distinguished ops that change the layout of their results.
  //
  // - linalg.reduce
  //
  std::string opName(operation->getName().getStringRef());
  if (layoutChangingOps.contains(opName)) {
    resultLayoutIsFree = LayoutIsFree(false);
  }

  for (LayoutIsFreeLattice* result : results) {
    propagateIfChanged(result, result->join(resultLayoutIsFree));
  }
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

}  // namespace heir
}  // namespace mlir
