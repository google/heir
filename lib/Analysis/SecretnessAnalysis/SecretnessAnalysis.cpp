#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/Casting.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE = "secretness-analysis"

namespace mlir {
namespace heir {

void SecretnessAnalysis::setToEntryState(SecretnessLattice* lattice) {
  auto value = lattice->getAnchor();
  bool secretness = isa<SecretTypeInterface>(value.getType());
  auto blockArg = dyn_cast<BlockArgument>(value);

  Operation* operation = nullptr;
  // Get defining operation for operand
  if (blockArg) {
    operation = blockArg.getOwner()->getParentOp();
  } else {
    operation = value.getDefiningOp();
  }

  // If operand is defined by a secret.generic operation,
  // check if operand is of secret type
  if (auto genericOp = dyn_cast<secret::GenericOp>(*operation)) {
    secretness = isa<secret::SecretType>(
        genericOp.getOperand(blockArg.getArgNumber()).getType());
  }

  // If operand is defined by a func.func operation,
  // check if the operand is either of secret type or annotated with
  // {secret.secret}
  if (auto funcOp = dyn_cast<func::FuncOp>(*operation)) {
    Type argType = funcOp.getArgumentTypes()[blockArg.getArgNumber()];
    // Check if it has secret-like type (secret, ciphertext, or shaped type
    // thereof)
    secretness = isa<SecretTypeInterface>(getElementTypeOrSelf(argType));

    // check if it is annotated as {secret.secret}
    UnitAttr attr = funcOp.getArgAttrOfType<UnitAttr>(
        blockArg.getArgNumber(), secret::SecretDialect::kArgSecretAttrName);
    if (attr) {
      secretness = true;
    }
  }

  propagateIfChanged(lattice, lattice->join(Secretness(secretness)));
}

LogicalResult SecretnessAnalysis::visitOperation(
    Operation* operation, ArrayRef<const SecretnessLattice*> operands,
    ArrayRef<SecretnessLattice*> results) {
  LDBG() << "Visiting operation " << operation->getName();
  auto resultSecretness = Secretness();
  bool isUninitializedOpFound = false;

  // Handle operations without operands (e.g. arith.constant)
  if (operands.empty()) {
    resultSecretness.setSecretness(false);
  }

  // Handle operations that have secret operands but no secret results
  if (isa<secret::RevealOp>(operation)) {
    resultSecretness.setSecretness(false);
    for (SecretnessLattice* result : results) {
      propagateIfChanged(result, result->join(resultSecretness));
    }
    return success();
  }

  LDBG() << "Secretness of operands and results: ";
  int i = 0;
  for (const SecretnessLattice* operand : operands) {
    const Secretness operandSecretness = operand->getValue();
    LDBG() << "o(" << i << ") = " << operandSecretness;
    if (!operandSecretness.isInitialized()) {
      // Keep record if operand is uninitialized
      isUninitializedOpFound = true;
    }
    resultSecretness = Secretness::join(resultSecretness, operandSecretness);
    if (resultSecretness.isInitialized() && resultSecretness.getSecretness())
      break;
    ++i;
  }
  LDBG() << "result = " << resultSecretness;

  // Uninitialized operand: "false" needs to be reverted to "unknown"
  // "secret" can remain, as "unknown + secret = secret"
  // As region-bearing ops are not yet supported in the secretness analysis
  // (except for control-flow, which the analysis framework handles directly),
  // we apply the same conservative logic if any regions are detected
  // TODO (#888): Handle region-bearing ops via visitNonControlFlowArguments
  if (isUninitializedOpFound || operation->getNumRegions()) {
    if (resultSecretness.isInitialized() && !resultSecretness.getSecretness()) {
      LDBG() << "for uninitialized operand, converting to unknown secretness";
      resultSecretness = Secretness();
    }
  }

  for (SecretnessLattice* result : results) {
    propagateIfChanged(result, result->join(resultSecretness));
  }
  return success();
}

void SecretnessAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const SecretnessLattice*> argumentLattices,
    ArrayRef<SecretnessLattice*> resultLattices) {
  auto callback = std::bind(&SecretnessAnalysis::propagateIfChangedWrapper,
                            this, std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<Secretness, SecretnessLattice>(
      call, argumentLattices, resultLattices, callback);
}

void annotateSecretness(Operation* top, DataFlowSolver* solver, bool verbose) {
  // Attribute "Printing" Helper
  auto getAttribute =
      [&](const SecretnessLattice* secretnessLattice) -> NamedAttribute {
    if (!secretnessLattice) {
      return {secret::SecretDialect::kArgMissingAttrName,
              UnitAttr::get(top->getContext())};
    }
    if (!secretnessLattice->getValue().isInitialized()) {
      return {secret::SecretDialect::kArgUnknownAttrName,
              UnitAttr::get(top->getContext())};
    }
    if (secretnessLattice->getValue().getSecretness()) {
      return {secret::SecretDialect::kArgSecretAttrName,
              UnitAttr::get(top->getContext())};
    }
    return {secret::SecretDialect::kArgPublicAttrName,
            UnitAttr::get(top->getContext())};
  };

  // Add an attribute to the operations to show determined secretness
  top->walk([&](Operation* op) {
    // Custom Handling for `func.func`, which uses special attributes
    if (auto func = llvm::dyn_cast<func::FuncOp>(op)) {
      if (func.isDeclaration()) {
        return;
      }
      // Arguments
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        auto arg = func.getArgument(i);
        // Do not annotate if already of type !secret.secret<..>
        if (!verbose && llvm::isa<secret::SecretType>(arg.getType())) continue;
        auto* secretnessLattice = solver->lookupState<SecretnessLattice>(arg);
        if (verbose || isSecret(secretnessLattice)) {
          auto attr = getAttribute(secretnessLattice);
          func.setArgAttr(i, attr.getName(), attr.getValue());
        }
      }

      // Results
      auto* ret = func.getFunctionBody().back().getTerminator();
      assert(ret->getNumOperands() == func.getFunctionType().getNumResults() &&
             "Number of returned values does not match function type");
      for (unsigned i = 0; i < func.getFunctionType().getNumResults(); ++i) {
        auto res = ret->getOpOperand(i).get();
        // Do not annotate if already of type !secret.secret<..>
        if (!verbose && llvm::isa<secret::SecretType>(res.getType())) continue;
        auto* secretnessLattice = solver->lookupState<SecretnessLattice>(res);
        if (verbose || isSecret(secretnessLattice)) {
          auto attr = getAttribute(secretnessLattice);
          func.setResultAttr(i, attr.getName(), attr.getValue());
        }
      }
    } else {
      // Default Handling for all other operations
      SmallVector<NamedAttribute, 1> attributes;
      bool isTerminator = op->hasTrait<OpTrait::IsTerminator>();
      if (isTerminator) {
        // Terminators (e.g., func.return, affine.yield) do not have mlir
        // op results, but do still have logical "results" (mlir operands)
        for (auto o : op->getOperands()) {
          // Do not annotate if already of type !secret.secret<..>
          if (!verbose && llvm::isa<secret::SecretType>(o.getType())) continue;
          auto* secretnessLattice = solver->lookupState<SecretnessLattice>(o);
          if (verbose || isSecret(secretnessLattice))
            attributes.append({getAttribute(secretnessLattice)});
        }
      } else {  // Non-Terminators, so consider op results
        for (auto o : op->getResults()) {
          // Do not annotate if already of type !secret.secret<..>
          if (!verbose && llvm::isa<secret::SecretType>(o.getType())) continue;
          auto* secretnessLattice = solver->lookupState<SecretnessLattice>(o);
          if (verbose || isSecret(secretnessLattice))
            attributes.append({getAttribute(secretnessLattice)});
        }
      }
      if (attributes.size() == 1) {
        // Do not annotate if already of type !secret.secret<..>
        if (verbose || !llvm::isa<secret::SecretType>(
                           isTerminator ? op->getOperand(0).getType()
                                        : op->getResult(0).getType()))
          op->setAttr(attributes[0].getName(), attributes[0].getValue());
      } else if (!attributes.empty()) {
        // Here, we emit also for !secret.secret<>) to preserve the mapping
        SmallVector<Attribute> dicts;
        for (auto a : attributes) {
          auto dict = DictionaryAttr::get(top->getContext(), a);
          dicts.push_back(dict);
        }
        auto arr = ArrayAttr::get(top->getContext(), dicts);
        op->setAttr("secretness", arr);
      }
    }

    return;
  });
}

bool isSecret(Value value, DataFlowSolver* solver) {
  auto* lattice = solver->lookupState<SecretnessLattice>(value);
  return isSecret(lattice);
}

bool isSecret(const SecretnessLattice* lattice) {
  if (!lattice) {
    return false;
  }
  if (!lattice->getValue().isInitialized()) {
    return false;
  }
  return lattice->getValue().getSecretness();
}

bool isSecret(ValueRange values, DataFlowSolver* solver) {
  if (values.empty()) {
    return false;
  }
  return std::all_of(values.begin(), values.end(),
                     [&](Value value) { return isSecret(value, solver); });
}

void getSecretOperands(Operation* op,
                       SmallVectorImpl<OpOperand*>& secretOperands,
                       DataFlowSolver* solver) {
  for (auto& operand : op->getOpOperands()) {
    if (isSecret(operand.get(), solver)) {
      secretOperands.push_back(&operand);
    }
  }
}

}  // namespace heir
}  // namespace mlir
