#include <optional>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#ifndef LIB_UTILS_ALLOCTOINPLACEUTILS_H_
#define LIB_UTILS_ALLOCTOINPLACEUTILS_H_

#include <algorithm>
#include <cassert>
#include <utility>

#include "lib/Utils/Tablegen/InPlaceOpInterface.h"
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/Liveness.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

// This library contains helpers for passes in HEIR that involve converting
// space-allocating operations to in-place variants. Cf., for example,
// lattigo-alloc-to-inplace.

#define DEBUG_TYPE "alloc-to-inplace"

namespace mlir {
namespace heir {

static std::optional<LevelState> getLevel(Value value, DataFlowSolver* solver) {
  auto* lattice = solver->lookupState<LevelLattice>(value);
  if (!lattice || !lattice->getValue().isInitialized()) {
    return std::nullopt;
  }
  auto latticeVal = lattice->getValue();
  if (!latticeVal.isInt()) {
    return std::nullopt;
  }
  return lattice->getValue().getInt();
}

// CallerProvidedStorageInfo provides an analysis of SSA values that
// can be reused for in-place operations that require the caller to pass
// in pre-allocated memory for the operation to use.
//
// This is used in backends like Lattigo, whose API for (e.g.) ciphertext
// addition takes as input a third operand to store the output (and do any
// internal work) in.
//
// This class distinguishes two types of Values in the IR:
//
// 1. Storage: the actual memory allocated for the value
// 2. Referring Value: the value that refers to the storage (e.g., the
//    returned SSA of an inplace operation)
//
// This class is similar to Disjoint-set data structure. Each Storage is the
// root, and all Referring Values are in its set.
//
// At the beginning CallerProvidedStorageInfo should be initialized based on
// current relation in program.
//
// During rewriting, when we find available Storage for an AllocOp, we replace
// it with an InPlaceOp and update this data structure by merging the Storage of
// the AllocOp to the available Storage.
//
// This allows a mix of AllocOp and InPlaceOp in input IR for the pass.
class CallerProvidedStorageInfo {
 public:
  CallerProvidedStorageInfo() = default;

  void addStorage(Value value) { storageToReferringValues[value] = {}; }

  void addReferringValue(Value storage, Value value) {
    storageToReferringValues[storage].push_back(value);
  }

 private:
  // maintenance should be called internally
  void removeStorage(Value value) { storageToReferringValues.erase(value); }

  void mergeStorage(Value from, Value to) {
    storageToReferringValues[to].reserve(storageToReferringValues[to].size() +
                                         storageToReferringValues[from].size());
    storageToReferringValues[to].insert(storageToReferringValues[to].end(),
                                        storageToReferringValues[from].begin(),
                                        storageToReferringValues[from].end());
    removeStorage(from);
  }

 public:
  // User API
  Value getStorageFromValue(Value value) const {
    LLVM_DEBUG(llvm::dbgs() << "getting storage from value " << value << "\n");
    for (auto& [storage, values] : storageToReferringValues) {
      if (value == storage) {
        LLVM_DEBUG(llvm::dbgs() << "found direct storage " << storage << "\n");
        return storage;
      }
      for (auto referringValue : values) {
        if (value == referringValue) {
          LLVM_DEBUG(llvm::dbgs()
                     << "found referring value " << referringValue << "\n");
          return storage;
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "did not find value!\n");
    return Value();
  }

  // Greedily use the first storage.
  //
  // This greedy policy is optimal in terms of memory usage in that
  // 1. All dead values for this operation are dead for later operations so
  // they are equivalent, which means the first dead value is enough.
  // 2. If we decide not to use inplace for this operation, but allocate a new
  // value, in the hope that later operation can benefit from the reserved value
  // of this decision. Later operation actually can always allocate a new value
  // so the memory usage is not affected by this operation's local decision.
  //
  // However, this might not be optimal in terms of cache-friendliness for
  // various accelerators. One basic optimization is to use the dead value that
  // is closest to the current operation in the block. But as we do not have the
  // information of the memory layout, we do not implement this optimization.
  Value getAvailableStorage(Operation* op, Liveness* liveness,
                            DataFlowSolver* solver) const {
    LLVM_DEBUG(llvm::dbgs()
               << "getAvailableStorage for op " << op->getName() << "\n");
    for (auto& [storage, values] : storageToReferringValues) {
      // storage and all referring values are dead
      if (solver) {
        auto opLevel = getLevel(op->getResult(0), solver);
        auto storageLevel = getLevel(storage, solver);
        if (!opLevel.has_value() || !storageLevel.has_value()) {
          continue;
        }
        if (opLevel.value() != storageLevel.value()) {
          continue;
        }
      }
      if (std::all_of(
              values.begin(), values.end(),
              [&](Value value) { return liveness->isDeadAfter(value, op); }) &&
          liveness->isDeadAfter(storage, op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "storage and all referring values are dead after "
                   << op->getName() << ": " << storage << "\n");
        return storage;
      }
    }
    return Value();
  }

  void replaceAllocWithInPlace(Operation* oldOp, Operation* newOp,
                               Value storage) {
    LLVM_DEBUG(llvm::dbgs()
               << "replacing alloc with inplace for " << oldOp->getName()
               << " at storage " << storage << "\n");
    // add newly created referring value
    for (auto result : newOp->getResults()) {
      addReferringValue(storage, result);
    }
    // remove storage of old op
    for (auto result : oldOp->getResults()) {
      mergeStorage(result, storage);
    }
  }

 private:
  DenseMap<Value, SmallVector<Value>> storageToReferringValues;
};

/// Initialize each func block's storage for the type T of interest.
template <typename T>
DenseMap<Block*, CallerProvidedStorageInfo>
initializeAllocToInPlaceBlockStorage(Operation* op) {
  DenseMap<Block*, CallerProvidedStorageInfo> blockToStorageInfo;
  op->walk([&](func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }
    for (auto& block : funcOp.getBody().getBlocks()) {
      auto& storageInfo = blockToStorageInfo[&block];
      // arguments are storages
      for (auto arg : block.getArguments()) {
        if (mlir::isa<T>(arg.getType())) {
          storageInfo.addStorage(arg);
        }
      }
      block.walk<WalkOrder::PreOrder>([&](Operation* op) {
        // inplace op will not allocate new memory, it produces referring
        // values
        if (auto inplaceOpInterface = mlir::dyn_cast<InPlaceOpInterface>(op)) {
          auto inplaceOperand =
              op->getOperand(inplaceOpInterface.getInPlaceOperandIndex());
          auto storage = storageInfo.getStorageFromValue(inplaceOperand);
          if (storage) {
            for (auto result : op->getResults()) {
              storageInfo.addReferringValue(storage, result);
            }
          }
        } else {
          // alloc op results are storages
          for (auto result : op->getResults()) {
            if (mlir::isa<T>(result.getType())) {
              storageInfo.addStorage(result);
            }
          }
        }
      });
    }
  });

  return blockToStorageInfo;
}

// OperandMutatedStorageInfo provides an analysis of SSA values that
// can be used for in-place operations that mutate an existing operand
// in-place.
//
// This is used in backends like OpenFHE, whose API for (e.g.) ciphertext
// addition mutates the LHS (or RHS, if a ciphertext) operand.
//
// This class distinguishes two types of Values in the IR:
//
// 1. Storage: the actual memory allocated for the value
// 2. Referring Value: the value that refers to the storage (e.g., the
//    result Value of an inplace operation)
//
// When replacing an op, this data structure should be updated so that the new
// result SSA value becomes a reference to the operand SSA value that was
// mutated.
class OperandMutatedStorageInfo {
 public:
  OperandMutatedStorageInfo() = default;

 private:
  void mergeStorage(Value from, Value to) {}

 public:
  /// Given a value, provide a canonical SSA value for the storage it
  /// represents. I.e., for in-place ops in this project, the in-place op itself
  /// will still return a new SSA value that refers to the mutated operand. This
  /// method takes a possibly referential Value and returns the source SSA
  /// value. If no reference is found the input value is returned unchanged.
  Value getStorageFromValue(Value value) const {
    if (valueToStorage.contains(value)) {
      Value storage = valueToStorage.at(value);
      LLVM_DEBUG(llvm::dbgs() << "Found storage " << storage << " for value "
                              << value << "\n");
      return storage;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Found no references for value " << value << "\n");
    return value;
  }

  Value getOldLivenessRef(Value value) const {
    if (newRefsToOldLiveness.contains(value)) {
      return newRefsToOldLiveness.at(value);
    }
    return value;
  }

  void replaceAllocWithInPlace(Operation* oldOp, Operation* newOp) {
    auto newInPlace = cast<InPlaceOpInterface>(newOp);
    Value mutated = newOp->getOperand(newInPlace.getInPlaceOperandIndex());
    assert(newOp->getNumResults() == 1);
    Value result = newOp->getResult(0);
    valueToStorage.insert(std::make_pair(result, mutated));
    LLVM_DEBUG(llvm::dbgs()
               << "Adding mapping for " << result << " to " << mutated << "\n");
    newRefsToOldLiveness.insert(std::make_pair(result, oldOp->getResult(0)));
  }

  bool isSafeToMutateInPlace(Operation* op, Value operand, Liveness* liveness) {
    Value storage = getStorageFromValue(operand);
    bool mutatedValueIsDead = liveness->isDeadAfter(operand, op);
    bool storageIsDead = liveness->isDeadAfter(storage, op);
    return mutatedValueIsDead && storageIsDead;
  }

 private:
  // Map from an SSA value to the canonical storage value.
  DenseMap<Value, Value> valueToStorage;

  // Map from results of newly inserted ops to the Values they replaced so as to
  // allow new ops to inherit old liveness
  DenseMap<Value, Value> newRefsToOldLiveness;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_ALLOCTOINPLACEUTILS_H_
