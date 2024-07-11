#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"

#include <llvm/Support/raw_ostream.h>

#include <optional>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project

namespace mlir {
namespace heir {

void SecretnessAnalysis::setToEntryState(SecretnessLattice *lattice) {
  auto operand = lattice->getPoint();
  bool isSecret = isa<secret::SecretType>(operand.getType());

  Operation *operation = nullptr;
  // Get defining operation for operand
  if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
    operation = blockArg.getOwner()->getParentOp();
  } else {
    operation = operand.getDefiningOp();
  }

  // If operand is defined by a secret.generic operation,
  // check if operand is of secret type
  if (auto genericOp = dyn_cast<secret::GenericOp>(*operation)) {
    if (OpOperand *genericOperand =
            genericOp.getOpOperandForBlockArgument(operand)) {
      isSecret = isa<secret::SecretType>(genericOperand->get().getType());
    }
  }
  propagateIfChanged(lattice, lattice->join(Secretness(isSecret)));
}

void SecretnessAnalysis::visitOperation(
    Operation *operation, ArrayRef<const SecretnessLattice *> operands,
    ArrayRef<SecretnessLattice *> results) {
  Secretness resultSecretness;
  for (const SecretnessLattice *operand : operands) {
    const Secretness operandSecretness = operand->getValue();
    if (operandSecretness.isInitialized()) {
      resultSecretness = Secretness::join(resultSecretness, operandSecretness);
      if (resultSecretness.getSecretness()) break;
    }
  }

  for (SecretnessLattice *result : results) {
    propagateIfChanged(result, result->join(resultSecretness));
  }
}

}  // namespace heir
}  // namespace mlir
