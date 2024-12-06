#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

void SecretnessAnalysis::setToEntryState(SecretnessLattice *lattice) {
  auto operand = lattice->getAnchor();
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

  // If operand is defined by a func.func operation,
  // check if the operand is either of secret type or annotated with
  // {secret.secret}
  if (auto funcOp = dyn_cast<func::FuncOp>(*operation)) {
    // identify which function argument the operand corresponds to
    auto blockArgs = funcOp.getBody().getArguments();
    int index = std::find(blockArgs.begin(), blockArgs.end(), operand) -
                blockArgs.begin();

    // Check if it has secret type
    isSecret = isa<secret::SecretType>(funcOp.getArgumentTypes()[index]);

    // check if it is annotated as {secret.secret}
    auto attrs = funcOp.getArgAttrs();
    if (attrs) {
      auto arr = attrs->getValue();
      if (auto dictattr = dyn_cast<DictionaryAttr>(arr[index])) {
        for (auto attr : dictattr) {
          isSecret =
              isSecret ||
              attr.getName() == secret::SecretDialect::kArgSecretAttrName.str();
          break;
        }
      }
    }
  }

  propagateIfChanged(lattice, lattice->join(Secretness(isSecret)));
}

LogicalResult SecretnessAnalysis::visitOperation(
    Operation *operation, ArrayRef<const SecretnessLattice *> operands,
    ArrayRef<SecretnessLattice *> results) {
  auto resultSecretness = Secretness();
  bool isUninitializedOpFound = false;

  // Handle operations without operands (e.g. arith.constant)
  if (operands.empty()) {
    resultSecretness.setSecretness(false);
  }

  for (const SecretnessLattice *operand : operands) {
    const Secretness operandSecretness = operand->getValue();
    if (!operandSecretness.isInitialized()) {
      // Keep record if operand is uninitialized
      isUninitializedOpFound = true;
    }
    resultSecretness = Secretness::join(resultSecretness, operandSecretness);
    if (resultSecretness.isInitialized() && resultSecretness.getSecretness())
      break;
  }

  // Uninitialized operand: "false" needs to be reverted to "unknown"
  // "secret" can remain, as "unknown + secret = secret"
  // As region-bearing ops are not yet supported in the secretness analysis
  // (except for control-flow, which the analysis framework handles directly),
  // we apply the same conservative logic if any regions are detected
  // TODO (#888): Handle region-bearing ops via visitNonControlFlowArguments
  if (isUninitializedOpFound || operation->getNumRegions()) {
    if (resultSecretness.isInitialized() && !resultSecretness.getSecretness()) {
      resultSecretness = Secretness();
    }
  }

  for (SecretnessLattice *result : results) {
    propagateIfChanged(result, result->join(resultSecretness));
  }
  return mlir::success();
}

void annotateSecretness(Operation *top, DataFlowSolver *solver) {
  // Add an attribute to the operations to show determined secretness
  top->walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      std::string name = op->getNumResults() == 1
                             ? "secretness"
                             : "result_" + std::to_string(i) + "_secretness";
      auto *secretnessLattice =
          solver->lookupState<SecretnessLattice>(op->getOpResult(i));
      if (!secretnessLattice) {
        op->setAttr(name, StringAttr::get(op->getContext(), "null"));
        return;
      }
      if (!secretnessLattice->getValue().isInitialized()) {
        op->setAttr(name, StringAttr::get(op->getContext(), ("unknown")));
        return;
      }
      op->setAttr(name,
                  BoolAttr::get(op->getContext(),
                                secretnessLattice->getValue().getSecretness()));
    }
    return;
  });
}

}  // namespace heir
}  // namespace mlir
