#include "include/Analysis/NoisePropagation/NoisePropagationAnalysis.h"

#include "include/Interfaces/NoiseInterfaces.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"       // from @llvm-project

#define DEBUG_TYPE "NoisePropagationAnalysis"

namespace mlir {
namespace heir {

void NoisePropagationAnalysis::visitOperation(
    Operation *op, ArrayRef<const VarianceLattice *> operands,
    ArrayRef<VarianceLattice *> results) {
  // If the lattice on any operand is unknown, bail out.
  if (llvm::any_of(operands, [](const VarianceLattice *lattice) {
        return !lattice->getValue().isKnown();
      })) {
    return;
  }

  auto noisePropagationOp = dyn_cast<NoisePropagationInterface>(op);
  if (!noisePropagationOp) return setAllToEntryStates(results);

  LLVM_DEBUG(llvm::dbgs() << "Propagating noise for " << *op << "\n");
  SmallVector<Variance> argRanges(llvm::map_range(
      operands, [](const VarianceLattice *val) { return val->getValue(); }));

  auto joinCallback = [&](Value value, const Variance &variance) {
    auto result = dyn_cast<OpResult>(value);
    if (!result) return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(llvm::dbgs() << "Inferred noise " << variance << "\n");
    VarianceLattice *lattice = results[result.getResultNumber()];
    Variance oldRange = lattice->getValue();
    ChangeResult changed = lattice->join(Variance{variance});

    // If the result is yielded, then the best we can do is check to see
    // if the op producing this value has deterministic noise. If so,
    // we can propagate that noise. Otherwise, we must assume the worst
    // case scenario of unknown noise.
    bool isYieldedResult = llvm::any_of(value.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    // FIXME: add DeterministicNoise trait
    if (isYieldedResult && oldRange.isKnown() &&
        !(lattice->getValue() == oldRange)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Non-deterministic noise-propagating op passed to a region "
             "terminator. Assuming loop result and marking noise unknown\n");
      changed |= lattice->join(Variance(std::nullopt));
    }
    propagateIfChanged(lattice, changed);
  };

  noisePropagationOp.inferResultNoise(argRanges, joinCallback);
}

}  // namespace heir
}  // namespace mlir
